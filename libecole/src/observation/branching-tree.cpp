#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <scip/scip.h>
#include <scip/struct_lp.h>
#include <xtensor/xview.hpp>

#include "ecole/observation/branching-tree.hpp"
#include "ecole/scip/model.hpp"
#include "ecole/scip/row.hpp"
#include "ecole/utility/unreachable.hpp"

namespace ecole::observation {

namespace {

/*********************
 *  Common helpers   *
 *********************/

using xmatrix = decltype(BranchingTreeObs::node_features);
using value_type = xmatrix::value_type;

using NodeFeatures = BranchingTreeObs::NodeFeatures;
using RowFeatures = BranchingTreeObs::RowFeatures;

value_type constexpr cste = 5.;
value_type constexpr nan = std::numeric_limits<value_type>::quiet_NaN();

SCIP_Real obj_l2_norm(SCIP* const scip) noexcept {
	auto const norm = SCIPgetObjNorm(scip);
	return norm > 0 ? norm : 1.;
}

/*******************************************
 *  Variable features extraction functions *
 *******************************************/

std::optional<SCIP_Real> upper_bound(SCIP* const scip, SCIP_COL* const col) noexcept {
	auto const ub_val = SCIPcolGetUb(col);
	if (SCIPisInfinity(scip, std::abs(ub_val))) {
		return {};
	}
	return ub_val;
}

std::optional<SCIP_Real> lower_bound(SCIP* const scip, SCIP_COL* const col) noexcept {
	auto const lb_val = SCIPcolGetLb(col);
	if (SCIPisInfinity(scip, std::abs(lb_val))) {
		return {};
	}
	return lb_val;
}

bool is_prim_sol_at_lb(SCIP* const scip, SCIP_COL* const col) noexcept {
	auto const lb_val = lower_bound(scip, col);
	if (lb_val) {
		return SCIPisEQ(scip, SCIPcolGetPrimsol(col), lb_val.value());
	}
	return false;
}

bool is_prim_sol_at_ub(SCIP* const scip, SCIP_COL* const col) noexcept {
	auto const ub_val = upper_bound(scip, col);
	if (ub_val) {
		return SCIPisEQ(scip, SCIPcolGetPrimsol(col), ub_val.value());
	}
	return false;
}

std::optional<SCIP_Real> best_sol_val(SCIP* const scip, SCIP_VAR* const var) noexcept {
	auto* const sol = SCIPgetBestSol(scip);
	if (sol != nullptr) {
		return SCIPgetSolVal(scip, sol, var);
	}
	return {};
}

std::optional<SCIP_Real> avg_sol(SCIP* const scip, SCIP_VAR* const var) noexcept {
	if (SCIPgetBestSol(scip) != nullptr) {
		return SCIPvarGetAvgSol(var);
	}
	return {};
}

std::optional<SCIP_Real> feas_frac(SCIP* const scip, SCIP_VAR* const var) noexcept {
	if (SCIPvarGetType(var) == SCIP_VARTYPE_CONTINUOUS) {
		return {};
	}
	return SCIPfeasFrac(scip, SCIPvarGetLPSol(var));
}

/** Convert an enum to its underlying index. */
template <typename E> constexpr auto idx(E e) {
	return static_cast<std::underlying_type_t<E>>(e);
}

template <typename Features>
void set_features_for_node(
	Features&& out,
	SCIP* const scip,
	SCIP_NODE* const node,
	value_type obj_norm,
	value_type n_lps) {
	out[idx(NodeFeatures::has_lower_bound)] = static_cast<value_type>(lower_bound(scip, col).has_value());
	out[idx(NodeFeatures::has_upper_bound)] = static_cast<value_type>(upper_bound(scip, col).has_value());
	out[idx(NodeFeatures::normed_reduced_cost)] = SCIPgetVarRedcost(scip, var) / obj_norm;
	out[idx(NodeFeatures::solution_value)] = SCIPvarGetLPSol(var);
	out[idx(NodeFeatures::solution_frac)] = feas_frac(scip, var).value_or(0.);
	out[idx(NodeFeatures::is_solution_at_lower_bound)] = static_cast<value_type>(is_prim_sol_at_lb(scip, col));
	out[idx(NodeFeatures::is_solution_at_upper_bound)] = static_cast<value_type>(is_prim_sol_at_ub(scip, col));
	out[idx(NodeFeatures::scaled_age)] = static_cast<value_type>(SCIPcolGetAge(col)) / (n_lps + cste);
	out[idx(NodeFeatures::incumbent_value)] = best_sol_val(scip, var).value_or(nan);
	out[idx(NodeFeatures::average_incumbent_value)] = avg_sol(scip, var).value_or(nan);
	// On-hot encoding
	out[idx(NodeFeatures::is_basis_lower)] = 0.;
	out[idx(NodeFeatures::is_basis_basic)] = 0.;
	out[idx(NodeFeatures::is_basis_upper)] = 0.;
	out[idx(NodeFeatures::is_basis_zero)] = 0.;
	switch (SCIPcolGetBasisStatus(col)) {
	case SCIP_BASESTAT_LOWER:
		out[idx(NodeFeatures::is_basis_lower)] = 1.;
		break;
	case SCIP_BASESTAT_BASIC:
		out[idx(NodeFeatures::is_basis_basic)] = 1.;
		break;
	case SCIP_BASESTAT_UPPER:
		out[idx(NodeFeatures::is_basis_upper)] = 1.;
		break;
	case SCIP_BASESTAT_ZERO:
		out[idx(NodeFeatures::is_basis_zero)] = 1.;
		break;
	default:
		utility::unreachable();
	}
}

void set_features_for_all_vars(xmatrix& out, scip::Model& model, bool const update_static) {
	auto* const scip = model.get_scip_ptr();

	// Contant reused in every iterations
	auto const n_lps = static_cast<value_type>(SCIPgetNLPs(scip));
	auto const obj_norm = obj_l2_norm(scip);

	auto const nodes = model.nodes();
	auto const n_nodes = nodes.size();
	for (std::size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
		auto* const node = nodes[node_idx];
		auto features = xt::row(out, static_cast<std::ptrdiff_t>(node_idx));
		set_features_for_node(features, scip, node, obj_norm, n_lps);
	}
}

/***************************************
 *  Row features extraction functions  *
 ***************************************/

SCIP_Real row_l2_norm(SCIP_ROW* const row) noexcept {
	auto const norm = SCIProwGetNorm(row);
	return norm > 0 ? norm : 1.;
}

SCIP_Real obj_cos_sim(SCIP* const scip, SCIP_ROW* const row) noexcept {
	auto const norm_prod = SCIProwGetNorm(row) * SCIPgetObjNorm(scip);
	if (SCIPisPositive(scip, norm_prod)) {
		return row->objprod / norm_prod;
	}
	return 0.;
}

/**
 * Number of inequality rows.
 *
 * Row are counted once per right hand side and once per left hand side.
 */
std::size_t n_ineq_rows(scip::Model& model) {
	auto* const scip = model.get_scip_ptr();
	std::size_t count = 0;
	for (auto* row : model.lp_rows()) {
		count += static_cast<std::size_t>(scip::get_unshifted_lhs(scip, row).has_value());
		count += static_cast<std::size_t>(scip::get_unshifted_rhs(scip, row).has_value());
	}
	return count;
}

template <typename Features>
void set_static_features_for_lhs_row(Features&& out, SCIP* const scip, SCIP_ROW* const row, value_type row_norm) {
	out[idx(RowFeatures::bias)] = -1. * scip::get_unshifted_lhs(scip, row).value() / row_norm;
	out[idx(RowFeatures::objective_cosine_similarity)] = -1 * obj_cos_sim(scip, row);
}

template <typename Features>
void set_static_features_for_rhs_row(Features&& out, SCIP* const scip, SCIP_ROW* const row, value_type row_norm) {
	out[idx(RowFeatures::bias)] = scip::get_unshifted_rhs(scip, row).value() / row_norm;
	out[idx(RowFeatures::objective_cosine_similarity)] = obj_cos_sim(scip, row);
}

template <typename Features>
void set_dynamic_features_for_lhs_row(
	Features&& out,
	SCIP* const scip,
	SCIP_ROW* const row,
	value_type row_norm,
	value_type obj_norm,
	value_type n_lps) {
	out[idx(RowFeatures::is_tight)] = static_cast<value_type>(scip::is_at_lhs(scip, row));
	out[idx(RowFeatures::dual_solution_value)] = -1. * SCIProwGetDualsol(row) / (row_norm * obj_norm);
	out[idx(RowFeatures::scaled_age)] = static_cast<value_type>(SCIProwGetAge(row)) / (n_lps + cste);
}

template <typename Features>
void set_dynamic_features_for_rhs_row(
	Features&& out,
	SCIP* const scip,
	SCIP_ROW* const row,
	value_type row_norm,
	value_type obj_norm,
	value_type n_lps) {
	out[idx(RowFeatures::is_tight)] = static_cast<value_type>(scip::is_at_rhs(scip, row));
	out[idx(RowFeatures::dual_solution_value)] = SCIProwGetDualsol(row) / (row_norm * obj_norm);
	out[idx(RowFeatures::scaled_age)] = static_cast<value_type>(SCIProwGetAge(row)) / (n_lps + cste);
}

auto set_features_for_all_rows(xmatrix& out, scip::Model& model, bool const update_static) {
	auto* const scip = model.get_scip_ptr();

	auto const n_lps = static_cast<value_type>(SCIPgetNLPs(scip));
	value_type const obj_norm = obj_l2_norm(scip);

	auto feat_row_idx = std::size_t{0};
	for (auto* const row : model.lp_rows()) {
		auto const row_norm = static_cast<value_type>(row_l2_norm(row));

		// Rows are counted once per rhs and once per lhs
		if (scip::get_unshifted_lhs(scip, row).has_value()) {
			auto features = xt::row(out, static_cast<std::ptrdiff_t>(feat_row_idx));
			if (update_static) {
				set_static_features_for_lhs_row(features, scip, row, row_norm);
			}
			set_dynamic_features_for_lhs_row(features, scip, row, row_norm, obj_norm, n_lps);
			feat_row_idx++;
		}
		if (scip::get_unshifted_rhs(scip, row).has_value()) {
			auto features = xt::row(out, static_cast<std::ptrdiff_t>(feat_row_idx));
			if (update_static) {
				set_static_features_for_rhs_row(features, scip, row, row_norm);
			}
			set_dynamic_features_for_rhs_row(features, scip, row, row_norm, obj_norm, n_lps);
			feat_row_idx++;
		}
	}
	assert(feat_row_idx == n_ineq_rows(model));
}

/****************************************
 *  Edge features extraction functions  *
 ****************************************/

/**
 * Number of non zero element in the constraint matrix.
 *
 * Row are counted once per right hand side and once per left hand side.
 */
auto matrix_nnz(scip::Model& model) {
	auto* const scip = model.get_scip_ptr();
	std::size_t nnz = 0;
	for (auto* row : model.lp_rows()) {
		auto const row_size = static_cast<std::size_t>(SCIProwGetNLPNonz(row));
		if (scip::get_unshifted_lhs(scip, row).has_value()) {
			nnz += row_size;
		}
		if (scip::get_unshifted_rhs(scip, row).has_value()) {
			nnz += row_size;
		}
	}
	return nnz;
}

utility::coo_matrix<value_type> extract_edge_features(scip::Model& model) {
	auto* const scip = model.get_scip_ptr();

	using coo_matrix = utility::coo_matrix<value_type>;
	auto const nnz = matrix_nnz(model);
	auto values = decltype(coo_matrix::values)::from_shape({nnz});
	auto indices = decltype(coo_matrix::indices)::from_shape({2, nnz});

	std::size_t i = 0;
	std::size_t j = 0;
	for (auto* const row : model.lp_rows()) {
		auto const row_norm = static_cast<value_type>(row_l2_norm(row));
		auto* const row_cols = SCIProwGetCols(row);
		auto const* const row_vals = SCIProwGetVals(row);
		auto const row_nnz = static_cast<std::size_t>(SCIProwGetNLPNonz(row));
		if (scip::get_unshifted_lhs(scip, row).has_value()) {
			for (std::size_t k = 0; k < row_nnz; ++k) {
				indices(0, j + k) = i;
				indices(1, j + k) = static_cast<std::size_t>(SCIPcolGetVarProbindex(row_cols[k]));
				values[j + k] = -row_vals[k] / row_norm;
			}
			j += row_nnz;
			i++;
		}
		if (scip::get_unshifted_rhs(scip, row).has_value()) {
			for (std::size_t k = 0; k < row_nnz; ++k) {
				indices(0, j + k) = i;
				indices(1, j + k) = static_cast<std::size_t>(SCIPcolGetVarProbindex(row_cols[k]));
				values[j + k] = row_vals[k] / row_norm;
			}
			j += row_nnz;
			i++;
		}
	}

	auto const n_rows = n_ineq_rows(model);
	// Change this here for nodes
	auto const n_vars = static_cast<std::size_t>(SCIPgetNVars(scip));
	return {values, indices, {n_rows, n_vars}};
}

auto is_on_root_node(scip::Model& model) -> bool {
	auto* const scip = model.get_scip_ptr();
	return SCIPgetCurrentNode(scip) == SCIPgetRootNode(scip);
}

auto extract_observation_fully(scip::Model& model) -> BranchingTreeObs {
	auto obs = BranchingTreeObs{
		// Change this here for nodes
		xmatrix::from_shape({model.nodes().size(), BranchingTreeObs::n_node_features}),
		xmatrix::from_shape({n_ineq_rows(model), BranchingTreeObs::n_row_features}),
		extract_edge_features(model),
	};
	set_features_for_all_vars(obs.node_features, model, true);
	set_features_for_all_rows(obs.row_features, model, true);
	return obs;
}

auto extract_observation_from_cache(scip::Model& model, BranchingTreeObs obs) -> BranchingTreeObs {
	set_features_for_all_vars(obs.node_features, model, false);
	set_features_for_all_rows(obs.row_features, model, false);
	return obs;
}

}  // namespace

/*************************************
 *  Observation extracting function  *
 *************************************/

auto BranchingTree::before_reset(scip::Model& /* model */) -> void {
	cache_computed = false;
}

auto BranchingTree::extract(scip::Model& model, bool /* done */) -> std::optional<BranchingTreeObs> {
	if (model.stage() == SCIP_STAGE_SOLVING) {
		if (use_cache) {
			if (is_on_root_node(model)) {
				the_cache = extract_observation_fully(model);
				cache_computed = true;
				return the_cache;
			}
			if (cache_computed) {
				return extract_observation_from_cache(model, the_cache);
			}
		}
		return extract_observation_fully(model);
	}
	return {};
}

}  // namespace ecole::observation

#include <algorithm>
#include <stdexcept>

#include <fmt/format.h>
#include <xtensor/xtensor.hpp>

#include "ecole/dynamics/node-selection.hpp"
#include "ecole/scip/model.hpp"
#include "ecole/scip/utils.hpp"

namespace ecole::dynamics {

NodeSelectionDynamics::NodeSelectionDynamics() noexcept {}

namespace {

auto action_set(scip::Model const& model) -> std::optional<xt::xtensor<std::size_t, 1>> {
	if (model.stage() != SCIP_STAGE_SOLVING) {
		return {};
	}

	auto const open_nodes = model.get_open_nodes();
	auto nodes_indexes = xt::xtensor<std::size_t, 1>::from_shape({std::get<0>(open_nodes).size() + std::get<1>(open_nodes).size() + std::get<2>(open_nodes).size()});
	auto const node_to_idx = [](auto const node) { return node->number; };
	std::transform(std::get<0>(open_nodes).begin(), std::get<0>(open_nodes).end(), nodes_indexes.begin(), node_to_idx);
	std::transform(std::get<1>(open_nodes).begin(), std::get<1>(open_nodes).end(), nodes_indexes.begin()+std::get<0>(open_nodes).size(), node_to_idx);
	std::transform(std::get<2>(open_nodes).begin(), std::get<2>(open_nodes).end(), nodes_indexes.begin()+std::get<0>(open_nodes).size()+std::get<1>(open_nodes).size(), node_to_idx);

	assert(nodes_indexes.size() > 0);
	return nodes_indexes;
}

/** Iterative solving until next LP branchrule call and return the action_set. */
template <typename FCall>
auto keep_solving_until_next_LP_callback(scip::Model& model, FCall& fcall)
	-> std::tuple<bool, NodeSelectionDynamics::ActionSet> {
	using Call = scip::callback::NodeSelectionCall;
	// While solving is not finished.
	while (fcall.has_value()) {
		// LP branchrule found, we give control back to the agent.
		// Assuming Branchrules are the only reverse callbacks.
		if (std::get<Call>(fcall.value()).where == Call::Where::Select) {
			return {false, action_set(model)};
		}
		// Otherwise keep looping, ignoring the callback.
		fcall = model.solve_iter_continue(SCIP_DIDNOTRUN);
	}
	// Solving is finished.
	return {true, {}};
}

}  // namespace

auto NodeSelectionDynamics::reset_dynamics(scip::Model& model) const -> std::tuple<bool, ActionSet> {
	auto fcall = model.solve_iter(scip::callback::NodeSelectionConstructor{});
	return keep_solving_until_next_LP_callback(model, fcall);
}

auto NodeSelectionDynamics::step_dynamics(scip::Model& model, Defaultable<std::size_t> maybe_node_idx) const
	-> std::tuple<bool, ActionSet> {
	// Default fallback to SCIP default branching
	SCIP_NODE* scip_result = nullptr;

	if (std::holds_alternative<std::size_t>(maybe_node_idx)) {
		auto const node_idx = std::get<std::size_t>(maybe_node_idx);
		auto const nodes = model.get_open_nodes();
		// Error handling
    for (int i = 0 ; scip_result == nullptr &&  i < std::get<0>(nodes).size() ; i++) {
				if (std::get<0>(nodes)[i]->number == node_idx) {
						scip_result = std::get<0>(nodes)[i];
						break;
				}
		}
		for (int i = 0 ; scip_result == nullptr &&  i < std::get<1>(nodes).size() ; i++) {
			if (std::get<1>(nodes)[i]->number == node_idx) {
				scip_result = std::get<1>(nodes)[i];
				break;
			}
		}
		for (int i = 0 ; scip_result == nullptr &&  i < std::get<2>(nodes).size() ; i++) {
			if (std::get<2>(nodes)[i]->number == node_idx) {
				scip_result = std::get<2>(nodes)[i];
				break;
			}
		}
		if (scip_result == nullptr) {
			throw std::invalid_argument{
				fmt::format("NodeSelection candidate index {} not found in open nodes.", node_idx)};
		}
	}

	// Looping until the next LP branchrule rule callback, if it exists.
	auto fcall = model.solve_iter_continue(scip_result);
	return keep_solving_until_next_LP_callback(model, fcall);
}

}  // namespace ecole::dynamics

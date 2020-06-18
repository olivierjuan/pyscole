#include <scip/scip.h>

#include "conftest.hpp"

ecole::scip::Model get_model() {
	auto model = ecole::scip::Model::from_file(problem_file);
	model.disable_cuts();
	model.disable_presolve();
	return model;
}

ecole::scip::Model get_solving_model() {
	auto model = get_model();
	model.solve_iter();
	return model;
}

#include "StdAfx.h"
#include "ModelTopologyBase.h"
#include "MathUtil.h"

void InitializeHiddenMarkovModelWithTopology(HiddenMarkovModel& model, ModelTopologyBase& topology, size_t symbols)
{
	model.states = topology.Create(true, model.symbols, model.GetTransitions(), model.GetProbabilities());			
	model.symbols = symbols;

	// Initialize B with uniform probabilities
	TMatrix logB(model.GetStates(), model.symbols, 1.0 / symbols);    
	logMatrix(logB);
	model.GetEmissions() = logB;
}
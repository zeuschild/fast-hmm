#include "StdAfx.h"
#include "ModelTopologyBase.h"
#include "MathUtil.h"

void InitializeHiddenMarkovModelWithTopology(HiddenMarkovModel& model, ModelTopologyBase& topology, size_t symbols)
{
	model.states = topology.Create(true, model.symbols, model.transitions, model.probabilities);			
	model.symbols = symbols;

	// Initialize B with uniform probabilities
	TMatrix logB(model.states, model.symbols, 1.0 / symbols);    
	logMatrix(logB);
	model.emissions = logB;
}
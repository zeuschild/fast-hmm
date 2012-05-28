#pragma once

#include "ImplConfig.h"

class HiddenMarkovModel
{	
	TRealVector probabilities;	
	TMatrix transitions;
	TMatrix emissions;	
	
public:
	std::size_t states;	
	std::size_t symbols;

	TRealVector& GetProbabilities();
	TMatrix& GetTransitions();
	TMatrix& GetEmissions();
	size_t GetStates() const;
	size_t GetSymbols() const;

	HiddenMarkovModel();
};

TReal EvaluateModel(HiddenMarkovModel& model, TSymbolVector& observations);
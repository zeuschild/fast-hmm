#pragma once

#include "ImplConfig.h"

class HiddenMarkovModel
{
public:
	std::size_t states;	
	std::size_t symbols;
	TRealVector probabilities;	
	TMatrix transitions;
	TMatrix emissions;	
	
	HiddenMarkovModel();
};

TReal EvaluateModel(HiddenMarkovModel& model, TSymbolVector& observations);
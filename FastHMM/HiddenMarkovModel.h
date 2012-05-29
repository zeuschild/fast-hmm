#pragma once

#include "ImplConfig.h"

class HiddenMarkovModel
{
public:
	size_t states;	
	size_t symbols;
	TRealVector probabilities;	
	TMatrix transitions;
	TMatrix emissions;	
	
	HiddenMarkovModel();
};

TReal EvaluateModel(const HiddenMarkovModel& model, const TSymbolVector& observations);
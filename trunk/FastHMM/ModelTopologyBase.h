#pragma once
#include "ImplConfig.h"
#include "HiddenMarkovModel.h"

class ModelTopologyBase
{
public:
	virtual size_t Create(bool logarithm, size_t symbols, TMatrix& transitionMatrix, TRealVector& initialState) = 0;
};

void InitializeHiddenMarkovModelWithTopology(HiddenMarkovModel& model, ModelTopologyBase& topology, size_t alpha);
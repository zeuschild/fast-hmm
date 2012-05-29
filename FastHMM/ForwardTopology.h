#pragma once

#include "ImplConfig.h"
#include "ModelTopologyBase.h"

class ForwardTopology : public ModelTopologyBase 
{	
	TRealVector pi;

public:
	ForwardTopology(int states, int deepness, bool random);
	
	size_t states;
	size_t deepness;
	bool random;	

	size_t Create(bool logarithm, size_t symbols, TMatrix& transitionMatrix, TRealVector& initialState);
};


#pragma once

#include "ImplConfig.h"

class ForwardTopology
{	
	TRealVector pi;

public:
	ForwardTopology(int states, int deepness, bool random);
	
	size_t states;
	size_t deepness;
	bool random;	

	size_t Create(bool logarithm, int symbols, TMatrix& transitionMatrix, TRealVector& initialState, TMatrix& emissions);
};


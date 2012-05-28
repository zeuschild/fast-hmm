#include "stdafx.h"
#include "HiddenMarkovModel.h"
#include "ForwardBackwardAlgorithm.h"

HiddenMarkovModel::HiddenMarkovModel()
	: states(0)
{
}

TRealVector& HiddenMarkovModel::GetProbabilities()
{
	return probabilities;
}

TMatrix& HiddenMarkovModel::GetTransitions()
{
	return transitions;
}

TMatrix& HiddenMarkovModel::GetEmissions()
{
	return emissions;
}

size_t HiddenMarkovModel::GetStates() const
{
	return states;
}

size_t HiddenMarkovModel::GetSymbols() const
{
	return symbols;
}


TReal EvaluateModel(HiddenMarkovModel& model, TSymbolVector& observations)
{
	if (observations.empty())
        return NegativeInfinity;
	    
	TMatrix lnFwd(observations.size(), model.GetStates());

    // Compute forward probabilities
    LogForward(model, observations, lnFwd);

	TReal logLikelihood = NegativeInfinity;
	for (size_t i = 0; i < model.GetStates(); i++)
		logLikelihood = LogSum(logLikelihood, lnFwd(observations.size() - 1, i));


    // Return the sequence probability
    return logLikelihood;
}
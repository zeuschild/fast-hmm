#include "stdafx.h"
#include "HiddenMarkovModel.h"
#include "ForwardBackwardAlgorithm.h"

HiddenMarkovModel::HiddenMarkovModel()
	: states(0)
{
}

/// <summary>
/// Evalua el modelo oculto de markov en la secuencia de observaciones y obtiene
/// la verosimilitud (en espacio logaritmico) con la cual explica la secuencia
/// </summary>
TReal EvaluateModel(HiddenMarkovModel& model, TSymbolVector& observations)
{
	if (observations.empty())
        return NegativeInfinity;
	    
	TMatrix lnFwd(observations.size(), model.states);

    // Compute forward probabilities
    LogForward(model, observations, lnFwd);

	TReal logLikelihood = NegativeInfinity;
	for (size_t i = 0; i < model.states; i++)
		logLikelihood = LogSum(logLikelihood, lnFwd(observations.size() - 1, i));


    // Return the sequence probability
    return logLikelihood;
}
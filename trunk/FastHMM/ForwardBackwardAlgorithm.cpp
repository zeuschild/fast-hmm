#include "stdafx.h"
#include "ForwardBackwardAlgorithm.h"

using namespace boost::math;

/// <summary>
///   Computes x + y without losing precision using ln(x) and ln(y).
/// </summary>
TReal LogSum(TReal lna, TReal lnc)
{
    if (lna == NegativeInfinity)
        return lnc;
    if (lnc == NegativeInfinity)
        return lna;

    if (lna > lnc)
        return lna + log1p<TReal>(exp(lnc - lna));
	else
		return lnc + log1p<TReal>(exp(lna - lnc));
}

/// <summary>
///   Computes Forward probabilities for a given hidden Markov model and a set of observations.
/// </summary>
void LogForward(HiddenMarkovModel& model, const TSymbolVector& observations, TMatrix& lnFwd)
{
	auto states = model.states;
	auto& logA = model.transitions;
	auto& logB = model.emissions;
	auto& logPi = model.probabilities;

	auto T = observations.size();

	// Ensures minimum requirements
	assert(lnFwd.size1() >= T);
	assert(lnFwd.size2() == states);

	lnFwd.clear();

	// 1. Initialization
	for (size_t i = 0; i < states; i++)
		lnFwd(0, i) = logPi[i] + logB(i, observations[0]);

    // 2. Induction
    for (size_t t = 1; t < T; t++)
    {
        auto x = observations[t];

        for (size_t i = 0; i < states; i++)
        {
			TReal sum = NegativeInfinity;
            for (size_t j = 0; j < states; j++)
                sum = LogSum(sum, lnFwd(t - 1, j) + logA(j, i));
            lnFwd(t, i) = sum + logB(i, x);
        }
    }
}

/// <summary>
///   Computes Backward probabilities for a given hidden Markov model and a set of observations.
/// </summary>
void LogBackward(HiddenMarkovModel& model, const TSymbolVector& observations, TMatrix& lnBwd)
{
	auto states = model.states;
	auto& logA = model.transitions;
	auto& logB = model.emissions;
	auto& logPi = model.probabilities;

	auto T = observations.size();

	// Ensures minimum requirements
	assert(lnBwd.size1() >= T);
	assert(lnBwd.size2() == states);

	lnBwd.clear();

	// 1. Initialization
	for (size_t i = 0; i < states; i++)
        lnBwd(T - 1, i) = 0;

	// 2. Induction
    for (long t = (long)(T - 2); t >= 0; t--)
    {
        for (size_t i = 0; i < states; i++)
        {
            TReal sum = NegativeInfinity;
            for (size_t j = 0; j < states; j++)
                sum = LogSum(sum, lnBwd(t + 1, j) + logA(i, j) + logB(j, observations[t + 1]));
            lnBwd(t, i) += sum;
        }
    }
}
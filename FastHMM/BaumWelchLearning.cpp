#include "stdafx.h"
#include "BaumWelchLearning.h"
#include "ForwardBackwardAlgorithm.h"

using namespace boost::math;

BaumWelchLearning::BaumWelchLearning(HiddenMarkovModel& _model, TReal _tolerance, int _maxIterations)
	: model(_model), tolerance(_tolerance), maxIterations(_maxIterations)
{	
}

/// <summary>
///   Checks if a model has converged given the likelihoods between two iterations
///   of the Baum-Welch algorithm and a criteria for convergence.
/// </summary>
bool BaumWelchLearning::HasConverged(TReal oldLogLikelihood, TReal newLogLikelihood, int currentIteration)
{
	// Update and verify stop criteria
    if (tolerance > 0)
    {
        // Stopping criteria is likelihood convergence
        TReal delta = abs(oldLogLikelihood - newLogLikelihood);
        if (delta <= tolerance)
            return true;

        if (maxIterations > 0)
        {
            // Maximum iterations should also be respected
            if (currentIteration >= maxIterations)
                return true;
        }
    }
    else
    {
        // Stopping criteria is number of iterations
        if (currentIteration == maxIterations)
            return true;
    }
	
    // Check if we have reached an invalid or perfectly separable answer
	if (isnan(newLogLikelihood) || isinf(newLogLikelihood))
    {
        return true;
    }

    return false;
}

/// <summary>
///   Runs the Baum-Welch learning algorithm for hidden Markov models.
/// </summary>
/// 
/// <param name="observations">The sequences of univariate or multivariate observations used to train the model.
///   Can be either of type double[] (for the univariate case) or double[][] for the
///   multivariate case.</param>
///   
/// <returns>
///   The average log-likelihood for the observations after the model has been trained.
/// </returns>
/// 
/// <remarks>
///   Learning problem. Given some training observation sequences O = {o1, o2, ..., oK}
///   and general structure of HMM (numbers of hidden and visible states), determine
///   HMM parameters M = (A, B, pi) that best fit training data.
/// </remarks>
TReal BaumWelchLearning::Run(const TObservationVector& observations)
{
	discreteObservations = &observations;

	if (observations.size() == 0)
		throw new std::exception("Observations vector must contain at least one observation");
	
	// Baum-Welch algorithm.

    // The Baum–Welch algorithm is a particular case of a generalized expectation-maximization
    // (GEM) algorithm. It can compute maximum likelihood estimates and posterior mode estimates
    // for the parameters (transition and emission probabilities) of an HMM, when given only
    // emissions as training data.

    // The algorithm has two steps:
    //  - Calculating the forward probability and the backward probability for each HMM state;
    //  - On the basis of this, determining the frequency of the transition-emission pair values
    //    and dividing it by the probability of the entire string. This amounts to calculating
    //    the expected count of the particular transition-emission pair. Each time a particular
    //    transition is found, the value of the quotient of the transition divided by the probability
    //    of the entire string goes up, and this value can then be made the new value of the transition.

	
    // Grab model information
	auto states = model.states;
	auto& logA = model.transitions;
	auto& logP = model.probabilities;
		
    // Initialize the algorithm
	auto N = observations.size();
    TReal logN = log((TReal)N);
	LogKsi.resize(N);
	LogGamma.resize(N);

	for (size_t i = 0; i < observations.size(); i++)
    {
		auto T = observations[i].size();

		LogKsi[i].resize(T, TMatrix(states, states, 0));
		LogGamma[i].resize(T, states);
		LogGamma[i].clear();
    }

	int iteration = 1;
    bool stop = false;

	auto itMax = std::max_element(observations.begin(), observations.end(), [](const TSymbolVector& x, const TSymbolVector& y) {return x.size() > y.size();});
	auto TMax = (*itMax).size();
    
	TMatrix lnFwd(TMax, states);
	TMatrix lnBwd(TMax, states);
    
    // Initialize the model log-likelihoods
	TReal newLogLikelihood = NegativeInfinity;
    TReal oldLogLikelihood = NegativeInfinity;

	do // Until convergence or max iterations is reached
    {
        // For each sequence in the observations input
		for (size_t i = 0; i < observations.size(); i++)
		{
			auto T = observations[i].size();
            auto& logGamma = LogGamma[i];


            // 1st step - Calculating the forward probability and the
            //            backward probability for each HMM state.
            ComputeForwardBackward(i, lnFwd, lnBwd);
			
            // 2nd step - Determining the frequency of the transition-emission pair values
            //            and dividing it by the probability of the entire string.

            // Calculate gamma values for next computations
            for (size_t t = 0; t < T; t++)
            {
				TReal lnsum = NegativeInfinity;
                for (size_t k = 0; k < states; k++)
                {
                    logGamma(t, k) = lnFwd(t, k) + lnBwd(t, k);
                    lnsum = LogSum(lnsum, logGamma(t, k));
                }

                // Normalize if different from zero
                if (lnsum != NegativeInfinity)
                    for (size_t k = 0; k < states; k++)
                        logGamma(t, k) = logGamma(t, k) - lnsum;
            }

            // Calculate ksi values for next computations
            ComputeKsi(i, lnFwd, lnBwd);

            // Compute log-likelihood for the given sequence
            newLogLikelihood = NegativeInfinity;
            for (size_t j = 0; j < states; j++)
                newLogLikelihood = LogSum(newLogLikelihood, lnFwd(T - 1, j));
        }


        // Average the likelihood for all sequences
		newLogLikelihood /= observations.size();


        // Check if the model has converged or if we should stop
        if (!HasConverged(oldLogLikelihood, newLogLikelihood, iteration))
        {
            // We haven't converged yet

            // 3. Continue with parameter re-estimation
            iteration++;
            oldLogLikelihood = newLogLikelihood;
            newLogLikelihood = NegativeInfinity;

            // 3.1 Re-estimation of initial state probabilities 
			for (size_t i = 0; i < logP.size(); i++)
            {
                TReal lnsum = NegativeInfinity;
				for (size_t k = 0; k < LogGamma.size(); k++)
                    lnsum = LogSum(lnsum, LogGamma[k](0, i));
                logP[i] = lnsum - logN;
            }

            // 3.2 Re-estimation of transition probabilities 
            for (size_t i = 0; i < states; i++)
            {
                for (size_t j = 0; j < states; j++)
                {
                    TReal lnnum = NegativeInfinity;
                    TReal lnden = NegativeInfinity;

					for (size_t k = 0; k < LogGamma.size(); k++)
                    {
						size_t T = observations[k].size();

                        for (size_t t = 0; t < T - 1; t++)
                        {
                            lnnum = LogSum(lnnum, LogKsi[k][t](i, j));
                            lnden = LogSum(lnden, LogGamma[k](t, i));
                        }
                    }

                    logA(i, j) = (lnnum == lnden) ? 0 : lnnum - lnden;
                }
            }

            // 3.3 Re-estimation of emission probabilities
            UpdateEmissions(); // discrete and continuous
        }
        else
        {
            stop = true; // The model has converged.
        }

    } while (!stop);


    // Returns the model average log-likelihood
    return newLogLikelihood;
}



/// <summary>
///   Computes the forward and backward probabilities matrices
///   for a given observation referenced by its index in the
///   input training data.
/// </summary>
/// 
/// <param name="index">The index of the observation in the input training data.</param>
/// <param name="lnFwd">Returns the computed forward probabilities matrix.</param>
/// <param name="lnBwd">Returns the computed backward probabilities matrix.</param>
void BaumWelchLearning::ComputeForwardBackward(size_t index, TMatrix& lnFwd, TMatrix& lnBwd)
{
	auto states = model.states;
	auto& obs = (*discreteObservations)[index];
    auto T = obs.size();

	assert(lnBwd.size1() >= T);
	assert(lnBwd.size2() == states);
	assert(lnFwd.size1() >= T);
	assert(lnFwd.size2() == states);

    LogForward(model, obs, lnFwd);
    LogBackward(model, obs, lnBwd);
}


/// <summary>
///   Updates the emission probability matrix.
/// </summary>
/// 
/// <remarks>
///   Implementations of this method should use the observations
///   in the training data and the Gamma probability matrix to
///   update the probability distributions of symbol emissions.
/// </remarks>
void BaumWelchLearning::UpdateEmissions()
{
	auto& B = model.emissions;
	auto states = model.states;
	auto symbols = model.symbols;

    for (size_t i = 0; i < states; i++)
    {
        for (size_t j = 0; j < symbols; j++)
        {
            TReal lnnum = NegativeInfinity;
            TReal lnden = NegativeInfinity;

			for (size_t k = 0; k < (*discreteObservations).size(); k++)
            {
				auto T = (*discreteObservations)[k].size();
                auto& gammak = LogGamma[k];

                for (size_t t = 0; t < T; t++)
                {
                    if ((*discreteObservations)[k][t] == j)
                        lnnum = LogSum(lnnum, gammak(t, i));
                    lnden = LogSum(lnden, gammak(t, i));
                }
            }

            // TODO: avoid locking a parameter in zero.
            B(i, j) = lnnum - lnden;
        }
    }
}



/// <summary>
///   Computes the ksi matrix of probabilities for a given observation
///   referenced by its index in the input training data.
/// </summary>
/// 
/// <param name="index">The index of the observation in the input training data.</param>
/// <param name="lnFwd">The matrix of forward probabilities for the observation.</param>
/// <param name="lnBwd">The matrix of backward probabilities for the observation.</param>
void BaumWelchLearning::ComputeKsi(size_t index, const TMatrix& lnFwd, const TMatrix& lnBwd)
{
	auto states = model.states;
	auto& logA = model.transitions;
	auto& logB = model.emissions;

    auto sequence = (*discreteObservations)[index];
	auto T = sequence.size();
    auto& logKsi = LogKsi[index];


    for (size_t t = 0; t < T - 1; t++)
    {
        TReal lnsum = NegativeInfinity;
        auto x = sequence[t + 1];

        for (size_t i = 0; i < states; i++)
        {
            for (size_t j = 0; j < states; j++)
            {
                logKsi[t](i, j) = lnFwd(t, i) + logA(i, j) + logB(j, x) + lnBwd(t + 1, j);
                lnsum = LogSum(lnsum, logKsi[t](i, j));
            }
        }

        for (size_t i = 0; i < states; i++)
            for (size_t j = 0; j < states; j++)
                logKsi[t](i, j) = logKsi[t](i, j) - lnsum;
    }
}
#pragma once

#include "ImplConfig.h"
#include "HiddenMarkovModel.h"
#include "ForwardBackwardAlgorithm.h"

class BaumWelchLearning
{
protected:
	HiddenMarkovModel& model;
	int maxIterations;
	TReal tolerance;

	std::vector<std::vector<TMatrix>> LogKsi;
	std::vector<TMatrix> LogGamma;

	const TObservationVector* discreteObservations;
	
	bool HasConverged(TReal oldLogLikelihood, TReal newLogLikelihood, int currentIteration);	
	void ComputeForwardBackward(size_t index, TMatrix& lnFwd, TMatrix& lnBwd);
	void ComputeKsi(size_t index, const TMatrix& lnFwd, const TMatrix& lnBwd);
	void UpdateEmissions();

public:
	double GetTolerance() const;
	int GetMaxIterations() const;
	double Run(const TObservationVector& observations);
	
	BaumWelchLearning(HiddenMarkovModel& model, TReal tolerance, int maxIterations);
};


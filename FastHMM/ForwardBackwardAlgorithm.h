#pragma once
#include "ImplConfig.h"
#include "HiddenMarkovModel.h"

TReal LogSum(TReal lna, TReal lnc);
void LogForward(const HiddenMarkovModel& model, const TSymbolVector& observations, TMatrix& lnFwd);		
void LogBackward(const HiddenMarkovModel& model, const TSymbolVector& observations, TMatrix& lnBwd);


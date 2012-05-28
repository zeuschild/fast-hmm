#pragma once
#include "ImplConfig.h"
#include "HiddenMarkovModel.h"

TReal LogSum(TReal lna, TReal lnc);
void LogForward(HiddenMarkovModel& model, const TSymbolVector& observations, TMatrix& lnFwd);		
void LogBackward(HiddenMarkovModel& model, const TSymbolVector& observations, TMatrix& lnBwd);


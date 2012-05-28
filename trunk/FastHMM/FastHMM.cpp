// FastHMM.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "BaumWelchLearning.h"
#include "ForwardTopology.h"

TObservationVector makeSamples(TSymbol* ini, size_t sampLen, size_t total)
{
	TObservationVector samples;
	for (auto it = ini; it != ini+total; it+=sampLen)
	{
		TSymbolVector sample;
		for (auto it2 = it; it2 != it+sampLen; it2++)
		{
			sample.push_back(*it2);
		}
		samples.push_back(sample);
	}
	return samples;
}

int main(int argc, char* argv[])
{
	auto model = HiddenMarkovModel();
	auto top = ForwardTopology(6, 6, false);
	model.symbols = 4;
	model.states = top.Create(true, model.symbols, model.GetTransitions(), model.GetProbabilities(), model.GetEmissions());		
	auto learning = BaumWelchLearning(model, 5e-4, 0);	
	TSymbol p1[] = 
	{ 
		0,1,2,3,0,1,2,
		0,1,1,3,0,1,2,
		0,1,2,1,0,1,2,
		0,1,2,3,1,1,2 
	};
	auto obs = makeSamples(p1, 7, 7*4);
	auto rtol = learning.Run(obs);
	TSymbol test[] = 
	{
		0,1,2,3,0,1,2,
		0,1,1,3,0,1,2,
		3,3,3,3,1,1,1,
		2,2,2,2,1,1,1,
	};
	auto test2 = makeSamples(test, 7, 7*4);
	auto r1 = EvaluateModel(model, test2[0]);
	auto r2 = EvaluateModel(model, test2[1]);
	auto r3 = EvaluateModel(model, test2[2]);
	auto r4 = EvaluateModel(model, test2[3]);
	return 0;
}


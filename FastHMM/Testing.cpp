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

void Testing()
{
	TSymbol p1[] = 
	{ 
		0,1,2,3,0,1,2,
		0,1,1,3,0,1,2,
		0,1,2,1,0,1,2,
		0,1,2,3,1,1,2 
	};
	auto obs = makeSamples(p1, 7, 7*4);

	auto model = HiddenMarkovModel();
	auto top = ForwardTopology(6, 6, false);
	InitializeHiddenMarkovModelWithTopology(model, top, 4);
	auto learning = BaumWelchLearning(model, 5e-4, 0);		
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
	const auto TOL = 1e-8;
	// valores de referencia con Accord.NET
	assert(fabs(r1-(-3.9364056688035625)) < TOL);
	assert(fabs(r2-(-5.0350078739166229)) < TOL);
	assert(fabs(r3-(-32602.524455560128)) < TOL);
	assert(fabs(r4-(-19547.858747514234)) < TOL);
	std::cout << "TESTING FINALIZADO CON EXITO" << std::endl;
}

/* PROGRAMA DE REFERENCIA
using Accord.Statistics.Models.Markov;
using Accord.Statistics.Models.Markov.Learning;
using Accord.Statistics.Models.Markov.Topology;
namespace testHMM
{
	class Program
	{
		static void Main(string[] args)
		{
			var topology = new Forward(6, 6, false);
			var model = new HiddenMarkovModel(topology, 4);
			var learning = new BaumWelchLearning(model);
			var observations = new[]{
				new[]{0,1,2,3,0,1,2},
				new[]{0,1,1,3,0,1,2},
				new[]{0,1,2,1,0,1,2},
				new[]{0,1,2,3,1,1,2},
			};
			learning.Tolerance = 5e-4;
			learning.Run(observations);
			var test = new[]{
				new []{0,1,2,3,0,1,2},
				new []{0,1,1,3,0,1,2},
				new []{3,3,3,3,1,1,1},
				new []{2,2,2,2,1,1,1},
			};
			var r1 = model.Evaluate(test[0]);
			var r2 = model.Evaluate(test[1]);
			var r3 = model.Evaluate(test[2]);
			var r4 = model.Evaluate(test[3]);
		}
	}
}
*/
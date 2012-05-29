// FastHMM.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "BaumWelchLearning.h"
#include "ForwardTopology.h"
#include "SamplesReader.h"

int main(int argc, char* argv[])
{
	SamplesReader reader;
	size_t alpha;
	TObservationVector pos, neg;
	reader.ReadSamples("C:\\Users\\Jairo\\Documents\\Visual Studio 2010\\Projects\\FastHMM\\x64\\Debug\\samples0.destino", pos, neg, &alpha);
	auto model = HiddenMarkovModel();
	auto top = ForwardTopology(6, 6, false);
	model.symbols = 4;
	model.states = top.Create(true, model.symbols, model.GetTransitions(), model.GetProbabilities(), model.GetEmissions());		
	auto learning = BaumWelchLearning(model, 5e-4, 0);
	auto tol = learning.Run(neg);
	std::cout << tol;
	return 0;
}


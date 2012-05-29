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
	auto symbols = std::string("ABCDEFGHIJKLMNOPQRSTUVWXYZ").length();
	reader.ReadSamples("C:\\Users\\Jairo\\Documents\\Visual Studio 2010\\Projects\\FastHMM\\x64\\Debug\\samples0.destino", pos, neg, &alpha);
	{
		auto model = HiddenMarkovModel();
		auto top = ForwardTopology(8, 8, false);
		InitializeHiddenMarkovModelWithTopology(model, top, symbols); 
		auto learning = BaumWelchLearning(model, 5e-4, 0);
		auto tol = learning.Run(pos);
	}
	{
		auto model = HiddenMarkovModel();
		auto top = ForwardTopology(8, 8, false);
		InitializeHiddenMarkovModelWithTopology(model, top, symbols); 
		auto learning = BaumWelchLearning(model, 5e-4, 0);
		auto tol = learning.Run(neg);
	}
	return 0;
}


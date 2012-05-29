// FastHMM.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "BaumWelchLearning.h"
#include "ForwardTopology.h"
#include "SamplesReader.h"
#include "HiddenMarkovModelExporter.h"

int main(int argc, char* argv[])
{		
	TObservationVector pos, neg;
	size_t symbols;
	SamplesReader reader;
	std::cout << "Cargando archivo de muestras." << std::endl;
	reader.ReadSamples("C:\\Users\\Jairo\\Documents\\Visual Studio 2010\\Projects\\FastHMM\\x64\\Debug\\samples0.destino", pos, neg, &symbols);	
	{
		std::cout << "Construyendo modelo Positivo" << std::endl;
		auto model = HiddenMarkovModel();
		auto top = ForwardTopology(8, 8, false);
		InitializeHiddenMarkovModelWithTopology(model, top, symbols); 
		auto learning = BaumWelchLearning(model, 5e-4, 0);
		auto tol = learning.Run(pos);
		std::cout << "Model obtenido con Likelihood = " << tol << std::endl;
		HiddenMarkovModelExporter::ExportPlainText(model, "positive_model.txt");
	}
	{
		std::cout << "Construyendo modelo Negativo" << std::endl;
		auto model = HiddenMarkovModel();
		auto top = ForwardTopology(8, 8, false);
		InitializeHiddenMarkovModelWithTopology(model, top, symbols); 
		auto learning = BaumWelchLearning(model, 5e-4, 0);
		auto tol = learning.Run(neg);
		std::cout << "Model obtenido con Likelihood = " << tol << std::endl;
		HiddenMarkovModelExporter::ExportPlainText(model, "negative_model.txt");
	}
	return 0;
}


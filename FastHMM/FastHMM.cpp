// FastHMM.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "BaumWelchLearning.h"
#include "ForwardTopology.h"
#include "SamplesReader.h"
#include "HiddenMarkovModelExporter.h"

using std::string;
using boost::lexical_cast;

void BuildHMM(HiddenMarkovModel& model, const TObservationVector& samples, size_t states, size_t alpha)
{
	auto tolerance = 5e-5;
	auto random = true;
	auto topology = ForwardTopology(states, states, random);
	InitializeHiddenMarkovModelWithTopology(model, topology, alpha);
	auto learning = BaumWelchLearning(model, tolerance, 0);
	auto likelihood = learning.Run(samples);
	std::cout << "Modelo obtenido con Likelihood = " << likelihood << std::endl;
}

void Create(string samplesFile, string pModelFile, string nModelFile, int states)
{
	TObservationVector pos, neg;
	size_t symbols;
	SamplesReader reader;
	auto model = HiddenMarkovModel();
	std::cout << "Cargando archivo de muestras." << std::endl;
	reader.ReadSamples(samplesFile, pos, neg, &symbols);
	{
		std::cout << "Construyendo modelo Positivo" << std::endl;		
		BuildHMM(model, pos, 8, symbols);
		HiddenMarkovModelExporter::ExportPlainText(model, pModelFile);
	}
	{
		std::cout << "Construyendo modelo Negativo" << std::endl;
		BuildHMM(model, pos, 8, symbols);
		HiddenMarkovModelExporter::ExportPlainText(model, nModelFile);
	}
}

int TestSample(const HiddenMarkovModel& pmodel, const HiddenMarkovModel& nmodel, const TSymbolVector& sample)
{
	auto l1 = EvaluateModel(pmodel, sample);
	auto l2 = EvaluateModel(nmodel, sample);
	return l1 > l2 ? 1 : 0;
}

void Test(string samplesFile, string pModelFile, string nModelFile, string reportFile)
{
	TObservationVector pos, neg;
	size_t symbols;
	SamplesReader reader;
	
	auto nmodel = HiddenMarkovModel();
	auto pmodel = HiddenMarkovModel();

	std::cout << "Cargando modelos." << std::endl;
	HiddenMarkovModelExporter::ImportPlainText(nmodel, nModelFile);
	HiddenMarkovModelExporter::ImportPlainText(pmodel, pModelFile);

	std::cout << "Cargando archivo de muestras." << std::endl;
	reader.ReadSamples(samplesFile, pos, neg, &symbols);

	std::cout << "Evaluando..." << std::endl;
	int pc = 0;
	int nc = 0;
	for(size_t i=0; i<pos.size(); i++)
	{
		auto r = TestSample(pmodel, nmodel, pos[i]);
		std::cout << "Evaluada P #" << i << " = " << r << std::endl;
		if(r == 1) pc++;
	}
	for(size_t i=0; i<neg.size(); i++)
	{
		auto r = TestSample(pmodel, nmodel, neg[i]);
		std::cout << "Evaluada N #" << i << " = " << r << std::endl;
		if(r == 0) nc++;
	}
}

int main(int argc, char* argv[])
{
	bool create = string(argv[0]) == "train";
	bool test = string(argv[0]) == "test";
	string filename1 = argv[1];
	string filename2 = argv[2];
	string filename3 = argv[3];
	if(create) 
	{
		int states = lexical_cast<int>(string(argv[4]));
		Create(filename1, filename2, filename3, states);
	}
	else if(test) 
	{
		string filename4 = argv[4];
		Test(filename1, filename2, filename3, filename4);
	}
	return 0;
}


// FastHMM.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"
#include "BaumWelchLearning.h"
#include "ForwardTopology.h"
#include "SamplesReader.h"
#include "HiddenMarkovModelExporter.h"

using namespace std;
using boost::lexical_cast;

// Construye un HMM con los parametros
void BuildHMM(HiddenMarkovModel& model, const TObservationVector& samples, size_t states, size_t alpha)
{
	auto tolerance = 5e-5;
	auto random = true;
	auto topology = ForwardTopology(states, states, random);
	InitializeHiddenMarkovModelWithTopology(model, topology, 26); // TODO: ARREGLAR EL ALPHA
	auto learning = BaumWelchLearning(model, tolerance, 0);
	auto likelihood = learning.Run(samples);
	std::cout << "Modelo obtenido con Likelihood = " << likelihood << std::endl;
}

// Crear model a partir de un archivo de muestras etiquetadas
void Create(string samplesFile, string pModelFile, string nModelFile, int states)
{
	cout << "Creacion de modelos" << endl;
	TObservationVector pos, neg;
	size_t symbols;
	SamplesReader reader;
	auto model = HiddenMarkovModel();
	cout << "Cargando archivo de muestras." << endl;
	reader.ReadSamples(samplesFile, pos, neg, &symbols);
	{
		cout << "Construyendo modelo Positivo" << endl;
		BuildHMM(model, pos, states*3, symbols);
		HiddenMarkovModelExporter::ExportPlainText(model, pModelFile);
	}
	{
		cout << "Construyendo modelo Negativo" << endl;
		BuildHMM(model, neg, states, symbols);
		HiddenMarkovModelExporter::ExportPlainText(model, nModelFile);
	}
}

// Prueba una muestra con el clasificador (dos HMM)
int TestSample(ofstream& report, int n, const HiddenMarkovModel& pmodel, const HiddenMarkovModel& nmodel, const TSymbolVector& sample)
{
	auto l1 = EvaluateModel(pmodel, sample);
	auto l2 = EvaluateModel(nmodel, sample);
	auto c = l1 > l2 ? 1 : 0;
	report << "Evaluation # " << n << " class: " << c << " p: " << l1 << " n: " << l2 << endl;
	return c;
}

// Prueba un clasificador de dos modelos usando un archivo de muestras
void Test(string samplesFile, string pModelFile, string nModelFile, string reportFile)
{
	TObservationVector pos, neg;
	size_t symbols;
	SamplesReader reader;
	
	auto nmodel = HiddenMarkovModel();
	auto pmodel = HiddenMarkovModel();

	cout << "Cargando modelos." << endl;
	HiddenMarkovModelExporter::ImportPlainText(nmodel, nModelFile);
	HiddenMarkovModelExporter::ImportPlainText(pmodel, pModelFile);

	cout << "Cargando archivo de muestras." << endl;
	reader.ReadSamples(samplesFile, pos, neg, &symbols);

	cout << "Evaluando..." << endl;
	int pc = 0;
	int nc = 0;
	ofstream report(reportFile);
	if(!report.is_open())
	{
		throw std::exception("Error with report file");
		return;
	}
	report << "Muestras Positivas" << endl;
	for(size_t i=0; i<pos.size(); i++)
	{
		auto r = TestSample(report, i, pmodel, nmodel, pos[i]);		
		if(r == 1) pc++;
	}
	report << "Muestras Negativas" << endl;
	for(size_t i=0; i<neg.size(); i++)
	{
		auto r = TestSample(report, i, pmodel, nmodel, neg[i]);		
		if(r == 0) nc++;
	}
	
	cout << "TP: " << pc << "/" << pos.size() << " TN: " << nc << "/" << neg.size() << endl;
	cout << "Total: " << (pos.size() + neg.size()) << " Total P: " << pos.size() << " Total N: " << neg.size() << endl;

	report << "TP: " << pc << "/" << pos.size() << " TN: " << nc << "/" << neg.size() << endl;
	report << "Total: " << (pos.size() + neg.size()) << " Total P: " << pos.size() << " Total N: " << neg.size() << endl;

	report.close();
}

int main(int argc, char* argv[])
{
	if(argc < 2) 
	{
		cout << "Para obtener ayuda ejecute \"FastHMM help\"" << endl;
		return 1;
	}
	bool create = string(argv[1]) == "train";
	bool test = string(argv[1]) == "test";	
	bool help = string(argv[1]) == "help";	
	if(create) 
	{
		if(argc != 6) 
		{
			cout << "Numero de argumentos incorrecto" << endl;
			return 1;
		}
		string filename1 = argv[2];
		string filename2 = argv[3];
		string filename3 = argv[4];
		int states = lexical_cast<int>(string(argv[5]));
		Create(filename1, filename2, filename3, states);
		return 0;
	}
	else if(test) 
	{
		if(argc != 6) 
		{
			cout << "Numero de argumentos incorrecto" << endl;
			return 1;
		}
		string filename1 = argv[2];
		string filename2 = argv[3];
		string filename3 = argv[4];
		string filename4 = argv[5];
		Test(filename1, filename2, filename3, filename4);
		return 0;
	}
	else if(help)
	{
		cout 
			<< "\tFastHMM {help|create|test} <options>" << endl
			<< "Options:" << endl
			<< endl
			<< "help" << endl
			<< "\tMuestra este mensaje de ayuda" << endl
			<< endl
			<< "train <samples> <pmodel> <nmodel> <nstates>" << endl
			<< "\tEntrena dos modelos desde las muestras etiquetadas positivas" << endl
			<< "\ty negativas en el archivo <samples>." << endl
			<< "\tGuarda los modelos en los archivos <pmodel> y <nmodel>" << endl
			<< endl
			<< "test <samples> <pmodel> <nmodel> <report>" << endl
			<< "\tEvalua los modelos en los archivos <pmodel> y <nmodel> sobre el archivo" << endl
			<< "\tde muestras realizando clasificacion binaria por comparacion de." << endl
			<< "\tverosimilitud. Los resultados se escriben en el archivo <report>" << endl
			<< endl
			;
		return 0;
	}
	else 
	{
		cout << "Opcion invalida: '" << argv[1] << "'" << endl;
		return 1;
	}
}


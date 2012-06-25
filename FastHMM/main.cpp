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
void BuildHMM(HiddenMarkovModel& model, const TObservationVector& samples, size_t states, size_t alpha, bool random)
{
	auto tolerance = 5e-5;	
	auto topology = ForwardTopology(states, states, random);
	InitializeHiddenMarkovModelWithTopology(model, topology, alpha);
	auto learning = BaumWelchLearning(model, tolerance, 0);
	auto likelihood = learning.Run(samples);
	std::cout << "Modelo obtenido con Log(Likelihood) = " << likelihood << std::endl;
}

// Crear modelos a partir de un archivo de muestras etiquetadas
void TrainSingle(string samplesFile, string pModelFile, string nModelFile, int states, bool random)
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
		BuildHMM(model, pos, states*3, symbols, random);
		HiddenMarkovModelExporter::ExportPlainText(model, pModelFile);
	}
	{
		cout << "Construyendo modelo Negativo" << endl;
		BuildHMM(model, neg, states, symbols, random);
		HiddenMarkovModelExporter::ExportPlainText(model, nModelFile);
	}
}

void TrainMultiple(string samplesFilename, string manifestFilename, int states, int count, bool random)
{
	ofstream manifest(manifestFilename);
	if(!manifest.is_open())
	{
		throw exception("No fue posible abrir el archivo de manifiesto");
	}
	manifest << "# Manifiesto de clasificador" << endl;
	manifest << "# Los siguientes archivos de modelos referenciados" << endl;
	manifest << "# p: Modelo para muestras positivas" << endl;
	manifest << "# n: Modelo para muestras negativas" << endl;
	string pmodelFilename;
	string nmodelFilename;
	for(int i=0; i<count; i++)
	{
		pmodelFilename = string("phmm-") + lexical_cast<string>(i) + ".hmm";
		nmodelFilename = string("nhmm-") + lexical_cast<string>(i) + ".hmm";
		TrainSingle(samplesFilename, pmodelFilename, nmodelFilename, states, random);
		manifest << "p " << pmodelFilename << endl;
		manifest << "n " << nmodelFilename << endl;
		cout << "Progreso global: modelo " << i << " (" << ((i+1)*100/count) << "%)" << endl;
	}
	manifest.close();
}

// Prueba una muestra con el clasificador (dos HMM)
int TestSample(ofstream& report, size_t n, const HiddenMarkovModel& pmodel, const HiddenMarkovModel& nmodel, const TSymbolVector& sample)
{
	auto l1 = EvaluateModel(pmodel, sample);
	auto l2 = EvaluateModel(nmodel, sample);
	auto c = l1 > l2 ? 1 : 0;
	report << "Evaluation # " << n << " class: " << c << " p: " << l1 << " n: " << l2 << endl;
	return c;
}


// Lee los modelos indicados en el manifiesto
void ReadManifest(string manifestFilename, vector<string>& pmodels, vector<string>& nmodels)
{
	ifstream manifest(manifestFilename);
	if(!manifest.is_open())
	{
		throw std::exception("Error with manifest file");
		return;
	}
	string line;
	while(!manifest.eof())
	{
		getline(manifest, line);
		boost::trim(line);
		if(line.size() < 3 || line[0] == '#') continue;
		// es un modelo positivo o negativo
		if(line[0] == 'p' && line[1] == ' ') 
		{
			pmodels.push_back(line.substr(2));
		}
		else if(line[0] == 'n' || line[1] == ' ')
		{
			nmodels.push_back(line.substr(2));
		}
		else
		{
			throw std::exception("Error with manifest file");
		}
	}
	manifest.close();
}

// Escribe los resultados de un experimento
void ReportMetric(ostream& report, int tp, int tn, int totalP, int totalN)
{
	auto fp = totalP - tp;
	auto fn = totalN - tn;
	auto acc = (tp + tn)/(float)(totalP + totalN);
	auto sens = tp/(float)totalP;
	auto spec = tn/(float)totalN;
	auto mcc = (tp*tn - fp*fn)/sqrt((float)(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));

	// informa resultado
	report << "True Positives: " << tp << ", True Negatives: " << tn << endl;	
	report << "Total Samples: " << (totalP + totalN) << ", Total P-samples: " << totalP << ", Total N-samples: " << totalN << endl;
	report << "Accuracy: " << acc << ", Sensitivity: " << sens << ", Specificity: " << spec << ", MCC: " << mcc << endl;
}

// Evalua un conjunto de modelos sobre un conjunto de muestras
void TestMultiple(string samplesFilename, string modelsManifestFilename, string reportFilename)
{
	TObservationVector pos, neg;
	vector<HiddenMarkovModel> pmodels;
	vector<HiddenMarkovModel> nmodels;
	size_t classifierCount;
	// Carga modelos del clasificador
	{
		cout << "Cargando manifiesto." << endl;
		vector<string> pModelFiles, nModelFiles;
		ReadManifest(modelsManifestFilename, pModelFiles, nModelFiles);
		if(pModelFiles.size() != nModelFiles.size()) 
		{
			throw exception("Se esperaba la misma cantidad de modelos positivos y negativos");
		}
		classifierCount = pModelFiles.size(); // la cantidad de clasificadores
		for(auto i = pModelFiles.begin(); i!=pModelFiles.end(); i++)
		{
			HiddenMarkovModel model;
			HiddenMarkovModelExporter::ImportPlainText(model, *i);
			pmodels.push_back(model);
			cout << "Modelo POSITIVO \"" << *i << "\" cargado." << endl;
		}
		for(auto i = nModelFiles.begin(); i!=nModelFiles.end(); i++)
		{
			HiddenMarkovModel model;
			HiddenMarkovModelExporter::ImportPlainText(model, *i);
			nmodels.push_back(model);
			cout << "Modelo NEGATIVO \"" << *i << "\" cargado." << endl;
		}
		cout << "Cargados " << (pmodels.size()+nmodels.size()) << " modelos en total" << endl;
	}	
	
	cout << "Cargando archivo de muestras." << endl;
	SamplesReader reader;
	
	size_t alpha;
	reader.ReadSamples(samplesFilename, pos, neg, &alpha);

	ofstream report(reportFilename);
	if(!report.is_open())
	{
		throw std::exception("Error abriendo el archivo de reporte");		
	}
	
	cout << "Evaluando..." << endl;
	
	// el umbral se fija en la mitad entera del numero de modelos	
	auto threshold = nmodels.size() / 2;
	int pc = 0, nc = 0;
	report << "Muestras Positivas" << endl;
	for(size_t i = 0; i < pos.size(); i++)
	{
		int answerCounter = 0;
		for(auto j=0; j < classifierCount; j++)
		{
			auto& pm = pmodels[j];
			auto& nm = nmodels[j];
			auto rj = TestSample(report, i, pm, nm, pos[i]);
			// cuenta los modelos que votan por "positiva"
			if(rj == 1) answerCounter++; 
		}		
		// si mas de la mitad de los votos son por "positiva"
		if(answerCounter > threshold) pc++; 
	}
	report << "Muestras Negativas" << endl;
	for(size_t i=0; i<neg.size(); i++)
	{
		int answerCounter = 0;
		for(auto j=0; j < classifierCount; j++)
		{
			auto& pm = pmodels[j];
			auto& nm = nmodels[j];
			auto r = TestSample(report, i, pm, nm, neg[i]);		
			// cuenta los modelos que votan por "negativa"
			if(r == 0) answerCounter++;
		}
		// si mas de la mitad de los votos son por "negativa"
		if(answerCounter > threshold) nc++;
	}

	// informa resultado
	ReportMetric(cout, pc, nc, (int)pos.size(), (int)neg.size());
	ReportMetric(report, pc, nc, (int)pos.size(), (int)neg.size());

	report.close();
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
	
	ReportMetric(cout, pc, nc, (int)pos.size(), (int)neg.size());
	ReportMetric(report, pc, nc, (int)pos.size(), (int)neg.size());

	report.close();
}

int main(int argc, char* argv[])
{
	try
	{
		if(argc < 2) 
		{
			cout << "Para obtener ayuda ejecute \"FastHMM help\"" << endl;
			return 1;
		}

		vector<string> arguments;
		for(int i=1; i<argc; i++)
		{
			arguments.push_back(argv[i]);
		}

		bool train_single = arguments[0] == "train_single";
		bool train_multiple = arguments[0] == "train_multiple";
		bool test_single = arguments[0] == "test_single";
		bool test_multiple = arguments[0] == "test_multiple";
		bool help = arguments[0] == "help";	

		if(help)
		{
			cout 
				<< "\tFastHMM {help|create|test} <options>" << endl
				<< "Options:" << endl
				<< endl
				<< "help" << endl
				<< "\tMuestra este mensaje de ayuda" << endl
				<< endl
				<< "train_single <samples> <pmodel> <nmodel> <nstates> [--seed=N|--no-random]" << endl
				<< "\tEntrena dos modelos desde las muestras etiquetadas positivas" << endl
				<< "\ty negativas en el archivo <samples>." << endl
				<< "\tGuarda los modelos en los archivos <pmodel> y <nmodel>" << endl
				<< endl
				<< "train_multiple <samples> <models-manifest> <count> <nstates> [--seed=N|--no-random]" << endl
				<< "\tEntrena a partir de las muestras del archivo <samples> varios modelos" << endl
				<< "\tque pueden ser utilizados como un comite de expertos" << endl
				<< endl
				<< "test_single <samples> <pmodel> <nmodel> <report>" << endl
				<< "\tEvalua los modelos en los archivos <pmodel> y <nmodel> sobre el archivo" << endl
				<< "\tde muestras realizando clasificacion binaria por comparacion de." << endl
				<< "\tverosimilitud. Los resultados se escriben en el archivo <report>" << endl
				<< endl
				<< "test_multiple <samples> <models-manifest> <report>" << endl
				<< "\tEvalua multiples modelos indicados en el archivo de manifiesto" << endl
				<< "\t<models-manifest> con las muestras en el archivo <samples>." << endl
				<< "\tEscribe los resultados en el archivo <report>" << endl
				<< endl
				<< ">> Shared options:" << endl				
				<< "\tLa opcion --seed=N le permite establecer N como la semilla de" << endl
				<< "\tgeneracion de numeros aleatorios. De esta manera puede generar" << endl
				<< "\tmodelos ocultos inicializados en orden aleatorio y conservar" << endl
				<< "\tel determinismo de los resultados de experimentacion" << endl
				;
		}
		else if(train_single || train_multiple) 
		{
			if(arguments.size() < 5) 
			{
				cout << "Numero de argumentos incorrecto" << endl;
				return 1;
			}
			string samplesFilename = arguments[1];			
			int states = lexical_cast<int>(arguments[4]);
			
			bool random = true;
			int customSeed = -1;
			if(arguments.size() == 6) 
			{
				if(arguments[5] == "--no-random")
				{
					random = false;
					cout << "NO usar inicializacion aleatoria" << endl;
				}
				if(boost::starts_with(arguments[5], "--seed="))
				{
					customSeed = lexical_cast<int>(arguments[5].substr(7));
					cout << "Semilla personalizada: " << customSeed << endl;
				}
			}
									
			auto t = customSeed < 0 ? time(NULL) : customSeed;	
			srand((unsigned)t);
			
			if(train_single)
			{
				string pModelFilename = arguments[2]; // solo para train single
				string nModelFilename = arguments[3]; // solo para train single
				TrainSingle(samplesFilename, pModelFilename, nModelFilename, states, random);	
			} 
			if(train_multiple)
			{
				int count = lexical_cast<int>(arguments[3]);
				string manifestFilename = arguments[2]; // solo para train multiple			
				TrainMultiple(samplesFilename, manifestFilename, states, count, random);
			}
		}	
		else if(test_single) 
		{
			if(arguments.size() != 5) 
			{
				cout << "Numero de argumentos incorrecto" << endl;
				return 1;
			}
			string samplesFilename = arguments[1];
			string pmodelFilename = arguments[2];
			string nmodelFilename = arguments[3];
			string reportFilename = arguments[4];			
			Test(samplesFilename, pmodelFilename, nmodelFilename, reportFilename);			
		}		
		else if(test_multiple) 
		{
			if(arguments.size() != 4) 
			{
				cout << "Numero de argumentos incorrecto" << endl;
				return 1;
			}
			string samplesFilename = arguments[1];
			string manifestFilename = arguments[2];
			string reportFilename = arguments[3];			
			TestMultiple(samplesFilename, manifestFilename, reportFilename);			
		}		
		else 
		{
			cout << "Opcion invalida: '" << argv[1] << "'" << endl;
			return 1;
		}
	}
	catch (const exception& e) 
	{
		string msg = e.what();
		cout << "Unexpected error: " << msg << endl;
		throw e;
	}
	return 0;
}


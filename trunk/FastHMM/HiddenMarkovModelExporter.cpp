#include "StdAfx.h"
#include "HiddenMarkovModelExporter.h"

using namespace std;
using namespace boost::algorithm;
using boost::lexical_cast;

ostream &operator << (ostream &os, const TMatrix &m)
{		
	for(size_t i = 0; i < m.size1(); i++)
	{
		for(size_t j = 0; j < m.size2(); j++)
		{
			auto val = m(i, j);
			os << lexical_cast<string>(val) << " ";
		}	
		os << endl;
	}
	return os;
}

ostream &operator << (ostream &os, const TRealVector &m)
{	
	for(size_t j = 0; j < m.size(); j++)
	{
		auto val = m[j];
		os << lexical_cast<string>(val) << " ";
	}
	os << endl;
	return os;
}

/// Escribe en un archivo de texto un HMM
void HiddenMarkovModelExporter::ExportPlainText(const HiddenMarkovModel& model, const std::string& filename)
{
	ofstream out(filename);
	out.precision(25);
	out << "# Hidden Markov Model" << endl;
	out << "# states symbols" << endl;
	out << model.states << " " << model.symbols << endl << endl;
	out << "# probabilities vector" << endl;
	out << model.probabilities << endl;
	out << "# transitions matrix" << endl;
	out << model.transitions << endl;
	out << "# emissions matrix" << endl;
	out << model.emissions << endl;
}
	
vector<string> splitBySpaces( string line )
{
	vector<string> splits;
	// divide la cadena por los espacios
	boost::split(splits, line, [](char s){return s==' ';});
	return splits;
}

/// Carga un archivo de texto con el modelo HMM
void HiddenMarkovModelExporter::ImportPlainText(HiddenMarkovModel& model, const std::string& filename)
{
	ifstream file(filename);
	if(!file.is_open()) 
	{
		throw exception("El archivo no pudo ser abierto");
	}
	string line;	
	int lineNumber=0;
	int counter;
	enum { header, probabilities, transitions, emissions, success } state;
	state = header;		
	while(!file.eof())
	{
		getline(file, line);
		trim(line);		
		lineNumber++;
		// ignora los comentarios
		if(line.length() > 0 && line[0] == '#') continue;
		if(state == header)
		{			
			if(line.empty()) 
			{
				state = probabilities;
				counter = 0;		
				continue;
			}
			auto splits = splitBySpaces(line);
			if(splits.size() != 2)
			{
				throw exception("Archivo de modelo invalido, cabecera mal formada");
			}
			model.states = lexical_cast<size_t>(splits[0]);
			model.symbols = lexical_cast<size_t>(splits[1]);			
		} 	
		else if(state == probabilities)
		{
			if(line.empty()) 
			{				
				state = transitions;
				counter = 0;		
				continue;
			}
			auto splits = splitBySpaces(line);
			auto width = splits.size();
			if(width == 0) throw exception("Archivo de modelo invalido, vector de probabilidades mal formada");			
			model.probabilities.resize(width);
			for(int i=0; i<width; i++)
			{
				auto val = lexical_cast<TReal>(splits[i]);
				model.probabilities[i] = val;
			}
		}
		else if(state == transitions)
		{			
			if(line.empty()) 
			{				
				state = emissions;
				counter = 0;
				continue;
			}
			auto splits = splitBySpaces(line);			
			auto width = splits.size();
			if(width == 0) throw exception("Archivo de modelo invalido, matriz de transiciones mal formada");			
			model.transitions.resize(counter+1, width);
			for(int i=0; i<width; i++)
			{
				auto val = lexical_cast<TReal>(splits[i]);
				model.transitions(counter, i) = val;
			}
			counter++;
		}
		else if(state == emissions)
		{			
			if(line.empty()) 
			{		
				state = success;
				break;
			}
			auto splits = splitBySpaces(line);			
			auto width = splits.size();
			if(width == 0) throw exception("Archivo de modelo invalido, matriz de emisiones mal formada");			
			model.emissions.resize(counter+1, width);
			for(int i=0; i<width; i++)
			{
				auto val = lexical_cast<TReal>(splits[i]);
				model.emissions(counter, i) = val;
			}
			counter++;
		}		
	}
	if(state != success) 
	{
		throw exception("Error leyendo el archivo de modelo");
	}
}

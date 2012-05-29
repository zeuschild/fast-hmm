#include "StdAfx.h"
#include "HiddenMarkovModelExporter.h"

using namespace std;
using namespace boost::algorithm;
using boost::lexical_cast;

ostream &operator << (ostream &os, const TMatrix &m)
{	
	ostream* o = &os;
	for(size_t i = 0; i < m.size1(); i++)
	{
		for(size_t j = 0; j < m.size2(); j++)
		{
			auto val = m(i, j);
			o = &((*o) << val << " ");
		}
		o = &((*o) << endl);
	}
	return *o;
}

ostream &operator << (ostream &os, const TRealVector &m)
{
	ostream* o = &os;
	for(size_t j = 0; j < m.size(); j++)
	{
		auto val = m[j];
		o = &((*o) << val << " ");
	}
	o = &((*o) << endl);	
	return *o;
}

void HiddenMarkovModelExporter::ExportPlainText(const HiddenMarkovModel& model, const std::string& filename)
{
	ofstream out(filename);
	out.precision(25);
	out << "# Hidden Markov Model" << endl;
	out << "# states symbols" << endl;
	out << model.states << " " << model.symbols << endl;
	out << "# probabilities vector" << endl;
	out << model.probabilities << endl;
	out << "# transitions matrix" << endl;
	out << model.transitions << endl;
	out << "# emissions matrix" << endl;
	out << model.emissions << endl;
}
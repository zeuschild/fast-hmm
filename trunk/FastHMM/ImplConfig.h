#pragma once

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <vector>

typedef double TReal;
typedef char TSymbol;
typedef std::vector<int> TIntVector;
typedef std::vector<TReal> TRealVector;
typedef std::vector<TSymbol> TSymbolVector;
typedef std::vector<TSymbolVector> TObservationVector;

typedef boost::numeric::ublas::matrix<TReal> TMatrix;

const TReal NegativeInfinity = -std::numeric_limits<TReal>::infinity();

inline void PrintMatrix(TMatrix& m)
{
	using namespace std;
	using boost::lexical_cast;
	cout << "Matrix" << endl;
	for(int i=0; i<m.size1(); i++)
	{
		for(int j=0; j<m.size2(); j++)
		{
			auto r = m(i, j);
			auto str = lexical_cast<string, TReal>(r);
			cout << str << " ";
		}
		cout << endl;
	}
}
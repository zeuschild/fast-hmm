#pragma once

#include "ImplConfig.h"

inline void logMatrix(TMatrix& m) 
{
	for (unsigned i = 0; i < m.size1 (); ++ i) 
	{
		for (unsigned j = 0; j < m.size2 (); ++ j)
		{
			auto r = m(i, j);
			r = log(r);
			m(i, j) = r;
		}
	}
}

inline void logVector(TRealVector& v)
{
	for (unsigned i = 0; i < v.size (); ++ i)
		v[i] = log(v[i]);
}


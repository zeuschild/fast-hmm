#include "StdAfx.h"
#include "ForwardTopology.h"

using std::min;

/// <summary>
///   Creates a new Forward topology for a given number of states.
/// </summary>
ForwardTopology::ForwardTopology(int _states, int _deepness, bool _random)
	: states(_states), deepness(_deepness), random(_random), pi(_states)
{	
	pi[0] = (TReal)1;
}


void logMatrix(TMatrix& m) 
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

void logVector(TRealVector& v)
{
	for (unsigned i = 0; i < v.size (); ++ i)
		v[i] = log(v[i]);
}

/// <summary>
///   Creates the state transitions matrix and the
///   initial state probabilities for this topology.
/// </summary>
size_t ForwardTopology::Create(bool logarithm, size_t symbols, TMatrix& transitionMatrix, TRealVector& initialState, TMatrix& emissions)
{
	boost::uniform_real<TReal> uni_dist(0,1);
	boost::mt19937_64 eng(42);
	boost::variate_generator<boost::mt19937_64, boost::uniform_real<TReal>> GenRand(eng, uni_dist);

    auto m = min(states, deepness);
    TMatrix A(states, states, 0);	
    if (random)
    {
        // Create A using random uniform distribution,
        //  without allowing backward transitions

        for (size_t i = 0; i < states; i++)
        {
            TReal sum = 0.0;
            for (size_t j = i; j < m; j++)
                sum += A(i, j) = GenRand();

            for (size_t j = i; j < m; j++)
                A(i, j) /= sum;
        }
    }
    else
    {
        // Create A using equal uniform probabilities,
        //   without allowing backward transitions.

        for (size_t i = 0; i < states; i++)
        {
            TReal d = 1.0 / min(m, states - i);
            for (size_t j = i; j < states && (j - i) < m; j++)
                A(i, j) = d;
        }
    }
    if (logarithm)
    {	
		logMatrix(A);
		logVector(pi);
    }
   
    transitionMatrix = A;
    initialState = pi;   

	///////////////////////////////////////////////////////////////////////////////////////

	// Initialize B with uniform probabilities
    TMatrix logB(states, symbols, 1.0 / symbols);    
	logMatrix(logB);
	emissions = logB;

    return states;
}
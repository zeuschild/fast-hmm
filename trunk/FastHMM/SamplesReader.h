#pragma once
#include "ImplConfig.h"
#include <vector>
#include <string>


class SamplesReader
{
public:
	void ReadSamples(std::string filename, TObservationVector& pos, TObservationVector& neg, size_t* alphabetLength);

	SamplesReader(void);
	~SamplesReader(void);
};


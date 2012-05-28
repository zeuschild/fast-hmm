#pragma once

#include <vector>
#include <string>
#include "ImplConfig.h"

class SamplesReader
{
public:
	void ReadSamples(std::string filename, TObservationVector& pos, TObservationVector& neg, size_t* alphabetLength);

	SamplesReader(void);
	~SamplesReader(void);
};


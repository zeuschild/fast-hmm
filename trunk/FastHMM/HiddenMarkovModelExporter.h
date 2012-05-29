#pragma once

#include <string>
#include "HiddenMarkovModel.h"

class HiddenMarkovModelExporter
{
public:
	static void ExportPlainText(const HiddenMarkovModel& model, const std::string& filename);
	static void ImportPlainText(HiddenMarkovModel& model, const std::string& filename);
};


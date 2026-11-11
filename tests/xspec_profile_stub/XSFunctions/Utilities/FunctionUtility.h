#pragma once
#include <map>
#include <string>
struct FunctionUtility {
    static const std::map<std::string, double>& getAllDbValues() {
        static std::map<std::string, double> values;
        return values;
    }
};

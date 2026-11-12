#pragma once
#include <map>
#include <string>
struct FunctionUtility {
    static std::map<std::string, std::string>& modelStrings() {
        static std::map<std::string, std::string> values;
        return values;
    }
    static const std::map<std::string, double>& getAllDbValues() {
        static std::map<std::string, double> values;
        return values;
    }
};

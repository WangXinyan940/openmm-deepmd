#include "DeepMDForce.h"
#include "internal/DeepMDForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>

using namespace DeepMDPlugin;
using namespace OpenMM;
using namespace std;

DeepMDForce::DeepMDForce(std::string filename, double scale) : dpfile(filename), usePeriodic(false), dpscale(scale) {
}

void DeepMDForce::setFile(std::string& filename) {
    dpfile = filename;
}

const string& DeepMDForce::getFile() const {
    return dpfile;
}

void DeepMDForce::setType(std::vector<int> type) {
    for(int i=0;i<type.size();i++){
        innertype.push_back(type[i]);
    }
}

const std::vector<int>& DeepMDForce::getType() const {
    return innertype;
}

ForceImpl* DeepMDForce::createImpl() const {
    return new DeepMDForceImpl(*this);
}

void DeepMDForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool DeepMDForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}

void DeepMDForce::setScale(double scale) {
    dpscale = scale;
}

double DeepMDForce::getScale() const {
    return dpscale;
}
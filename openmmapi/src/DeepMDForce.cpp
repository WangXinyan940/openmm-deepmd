#include "DeepMDForce.h"
#include "internal/DeepMDForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <fstream>

using namespace DeepMDPlugin;
using namespace OpenMM;
using namespace std;

DeepMDForce::DeepMDForce(const std::string& filename) : file(filename), usePeriodic(false) {
}

void setFile(std::string& filename) {
    file = filename;
}

const string& DeepMDForce::getFile() const {
    return file;
}

void setMask(std::vector<int> mask){
    for(int i=0;i<mask.size();i++){
        innermask.push_back(mask[i]);
    }
}

const std::vector<int>& getMask() const {
    return innermask&;
}

void setType(std::vector<int> type) {
    for(int i=0;i<type.size();i++){
        innertype.push_back(type[i]);
    }
}

const std::vector<int>& getType() const {
    return innertype&;
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
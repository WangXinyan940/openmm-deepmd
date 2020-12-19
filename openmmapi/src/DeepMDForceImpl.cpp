#include "internal/DeepMDForceImpl.h"
#include "DeepMDKernels.h"
#include "deepmd/NNPInter.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace DeepMDPlugin;
using namespace OpenMM;
using namespace std;

DeepMDForceImpl::DeepMDForceImpl(const DeepMDForce& owner) : owner(owner) {
}

DeepMDForceImpl::~DeepMDForceImpl() {
}

void DeepMDForceImpl::initialize(ContextImpl& context) {
    // Load deepmd-kit model from the file.
    NNPInter model(owner.getFile());

    // Create the kernel.
    kernel = context.getPlatform().createKernel(CalcDeepMDForceKernel::Name(), context);
    kernel.getAs<CalcDeepMDForceKernel>().initialize(context.getSystem(), owner, model);
}

double DeepMDForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcDeepMDForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> DeepMDForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcDeepMDForceKernel::Name());
    return names;
}
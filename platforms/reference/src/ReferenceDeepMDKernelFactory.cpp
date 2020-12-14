#include "ReferenceDeepMDKernelFactory.h"
#include "ReferenceDeepMDKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <vector>

using namespace DeepMDPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    int argc = 0;
    vector<char**> argv = {NULL};
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferenceDeepMDKernelFactory* factory = new ReferenceDeepMDKernelFactory();
            platform.registerKernelFactory(CalcDeepMDForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerDeepMDReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferenceDeepMDKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    ReferencePlatform::PlatformData& data = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcDeepMDForceKernel::Name())
        return new ReferenceCalcDeepMDForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
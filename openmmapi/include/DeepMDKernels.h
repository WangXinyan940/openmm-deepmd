#ifndef DEEPMD_KERNELS_H_
#define DEEPMD_KERNELS_H_

#include "DEEPMDForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "NNPInter.h"
#include <c_api.h>
#include <string>

namespace DeepMDPlugin {

/**
 * This kernel is invoked by DeepMDForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcDeepMDForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcDeepMDForce";
    }
    CalcDeepMDForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system         the System this kernel will be applied to
     * @param force          the DeepMDForce this kernel will be used for
     * @param model          the DeepMD-kit model
     */
    virtual void initialize(const OpenMM::System& system, const DeepMDForce& force, const NNPInter& model) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

} // namespace DeepMDPlugin

#endif /*DEEPMD_KERNELS_H_*/
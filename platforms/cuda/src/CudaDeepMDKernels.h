#ifndef CUDA_DEEPMD_KERNELS_H_
#define CUDA_DEEPMD_KERNELS_H_

#include "deepmd/NNPInter.h"
#include "DeepMDKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>
#include <string>

namespace DeepMDPlugin {

/**
 * This kernel is invoked by DeepMDForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcDeepMDForceKernel : public CalcDeepMDForceKernel {
public:
    CudaCalcDeepMDForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu) :
            CalcDeepMDForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    }
    ~CudaCalcDeepMDForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system         the System this kernel will be applied to
     * @param force          the DeepMDForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const DeepMDForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    NNPInter deepmodel;
    std::vector<int> types;
    bool hasInitializedKernel;
    OpenMM::CudaContext& cu;
    bool usePeriodic;
    OpenMM::CudaArray networkForces;
    CUfunction addForcesKernel;
    double dpscale;
};

} // namespace DeepMDPlugin

#endif /*CUDA_DEEPMD_KERNELS_H_*/
#ifndef REFERENCE_DEEPMD_KERNELS_H_
#define REFERENCE_DEEPMD_KERNELS_H_

#include "deepmd/NNPInter.h"
#include "deepmd/common.h"
#include "DeepMDKernels.h"
#include "openmm/Platform.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include <vector>
#include <set>
#include <iostream>

namespace DeepMDPlugin {

/**
 * This kernel is invoked by DeepMDForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcDeepMDForceKernel : public CalcDeepMDForceKernel {
public:
    ReferenceCalcDeepMDForceKernel(std::string name, const OpenMM::Platform& platform) : CalcDeepMDForceKernel(name, platform) {
    }
    ~ReferenceCalcDeepMDForceKernel();
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
    double rcut;
    std::vector<int> mask;
    std::vector<int> types;
    bool doubleModel;
    std::vector<float> positions, boxVectors;
    bool usePeriodic;
    OpenMM::NeighborList neighborList;
    vector<set<int>> ex;
};

} // namespace DeepMDPlugin

#endif /*REFERENCE_DEEPMD_KERNELS_H_*/
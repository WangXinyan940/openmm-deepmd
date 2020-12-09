#ifndef OPENMM_REFERENCE_DEEPMD_KERNEL_FACTORY_H_
#define OPENMM_REFERENCE_DEEPMD_KERNEL_FACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the reference implementation of the DeepMD plugin.
 */

class ReferenceDeepMDKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_REFERENCE_DEEPMD_KERNEL_FACTORY_H_*/
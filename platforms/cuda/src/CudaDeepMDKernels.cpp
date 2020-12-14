#include "CudaDeepMDKernels.h"
#include "CudaDeepMDKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

using namespace DeepMDPlugin;
using namespace OpenMM;
using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
typedef double ENERGYTYPE;
#else
typedef float VALUETYPE;
typedef double ENERGYTYPE;
#endif

CudaCalcDeepMDForceKernel::~CudaCalcDeepMDForceKernel() {
}

void CudaCalcDeepMDForceKernel::initialize(const System& system, const DeepMDForce& force, const NNPInter& model) {
    
    int numParticles = system.getNumParticles();
    // hold model
    NNPInter deepmodel = model;

    // create input tensors
    mask = force.getMask();
    types = force.getType();
    doubleModel = force.useDoublePrecision();

    // Inititalize CUDA objects.
    map<string, string> defines;
    #ifdef HIGH_PREC
        defines["FORCES_TYPE"] = "double";
        networkForces.initialize(cu, 3*numParticles, sizeof(double), "networkForces");
    #else
        defines["FORCES_TYPE"] = "float";
        networkForces.initialize(cu, 3*numParticles, sizeof(float), "networkForces");
    #endif
    CUmodule module = cu.createModule(CudaDeepMDKernelSources::deepMDForce, defines);
    addForcesKernel = cu.getKernel(module, "addForces");
}

double CudaCalcDeepMDForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3> pos;
    context.getPositions(pos);
    int numParticles = cu.getNumAtoms();
    
    vector<VALUETYPE> positions;
    for (int i = 0; i < mask.size(); i++) {
        positions.push_back(pos[mask[i]][0]);
        positions.push_back(pos[mask[i]][1]);
        positions.push_back(pos[mask[i]][2]);
    }

    vector<VALUETYPE> boxVectors(9,0);
    if (usePeriodic) {
        Vec3 box[3];
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                boxVectors.push_back(box[i][j]);
    } else {
        boxVectors[0] = 9999.9;
        boxVectors[4] = 9999.9;
        boxVectors[8] = 9999.9;
    }
    
    // run model
    vector<VALUETYPE> force_tmp(positions.size()*3, 0);
    vector<VALUETYPE> virial(9,0);
    ENERGYTYPE ener = 0;
    deepmodel.compute(ener, force_tmp, virial, positions, types, boxVectors);

    double energy = 0.0;
    if (includeEnergy) {
        energy = ener;
    }
    if (includeForces) {
        if (cu.getUseDoublePrecision()){
            vector<double> data(3*pos.size(),0);
        } else {
            vector<float> data(3*pos.size(),0);
        }
        for(int i=0;i<mask.size();i++){
            int p = mask[i];
            for(int j=0;j<3;j++){
                data[3*p+j] = force_tmp[3*i+j];
            }
        }
        networkForces.upload(data);
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&networkForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }
    return energy;
}
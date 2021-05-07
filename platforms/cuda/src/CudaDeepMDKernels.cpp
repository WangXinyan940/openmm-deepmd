#include "CudaDeepMDKernels.h"
#include "CudaDeepMDKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <iostream>

using namespace DeepMDPlugin;
using namespace OpenMM;
using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE2;
#else
typedef float VALUETYPE2;
#endif

CudaCalcDeepMDForceKernel::~CudaCalcDeepMDForceKernel() {
}

void CudaCalcDeepMDForceKernel::initialize(const System& system, const DeepMDForce& force) {
    

    int numParticles = system.getNumParticles();
    // hold model
    NNPInter model(force.getFile());
    deepmodel = model;

    // create input tensors
    types = force.getType();

    // Inititalize CUDA objects.
    map<string, string> defines;
    #ifdef HIGH_PREC
        cout << "High Prec" << endl;
        defines["FORCES_TYPE"] = "double";
        int networkForcesSize = sizeof(double);
    #else
        cout << "Low Prec" << endl;
        defines["FORCES_TYPE"] = "float";
        int networkForcesSize = sizeof(float);
    #endif
    dpscale = force.getScale();
    cu.setAsCurrent();
    networkForces.initialize(cu, 3*numParticles, networkForcesSize, "networkForces");
    CUmodule module = cu.createModule(CudaDeepMDKernelSources::deepMDForce, defines);
    addForcesKernel = cu.getKernel(module, "addForces");
}

double CudaCalcDeepMDForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    
    vector<Vec3> pos;
    context.getPositions(pos);
    int numParticles = cu.getNumAtoms();
    
    vector<VALUETYPE2> positions(3*numParticles,0.0);
    for (int i = 0; i < numParticles; i++) {
        // nm to angstrom
        positions[3*i] = pos[i][0]*10;
        positions[3*i+1] = pos[i][1]*10;
        positions[3*i+2] = pos[i][2]*10;
    }
    // cout << "Position loaded" << endl;

    vector<VALUETYPE2> boxVectors(9,0);
    if (usePeriodic) {
        Vec3 box[3];
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                boxVectors[i*3+j] = box[i][j]*10; // nm to angstrom
    } else {
        boxVectors[0] = 999.9;
        boxVectors[4] = 999.9;
        boxVectors[8] = 999.9;
    }
    // cout << "Box loaded" << endl;
    
    // run model
    vector<VALUETYPE2> force_tmp(positions.size(), 0);
    vector<VALUETYPE2> virial(9,0);
    double ener = 0;
    deepmodel.compute(ener, force_tmp, virial, positions, types, boxVectors);
    // cout << "Got result" << endl;

    double energy = 0.0;
    if (includeEnergy) {
        energy = ener*96*dpscale; // ev to kj/mol
    }
    if (includeForces) {
        vector<VALUETYPE2> data(3*pos.size(),0);
        for(int i=0;i<numParticles;i++){
            for(int j=0;j<3;j++){
                data[3*i+j] = force_tmp[3*i+j]*dpscale;
            }
        }
        networkForces.upload(data);
        int paddedNumAtoms = cu.getPaddedNumAtoms();

        void* args[] = {&networkForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }
    return energy;
}
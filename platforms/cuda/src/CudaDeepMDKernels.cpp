#include "CudaDeepMDKernels.h"
#include "CudaDeepMDKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <iostream>

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

    // Inititalize CUDA objects.
    map<string, string> defines;
    #ifdef HIGH_PREC
        cout << "High Prec" << endl;
        defines["FORCES_TYPE"] = "double";
        networkForces.initialize(cu, 3*numParticles, sizeof(double), "networkForces");
    #else
        cout << "Low Prec" << endl;
        defines["FORCES_TYPE"] = "double";
        networkForces.initialize(cu, 3*numParticles, sizeof(double), "networkForces");
    #endif
    CUmodule module = cu.createModule(CudaDeepMDKernelSources::deepMDForce, defines);
    addForcesKernel = cu.getKernel(module, "addForces");
}

double CudaCalcDeepMDForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3> pos;
    context.getPositions(pos);
    int numParticles = cu.getNumAtoms();

    cout << "Goes in" << endl;
    
    vector<VALUETYPE> positions;
    for (int i = 0; i < mask.size(); i++) {
        positions.push_back(pos[mask[i]][0]*10);
        positions.push_back(pos[mask[i]][1]*10);
        positions.push_back(pos[mask[i]][2]*10);
    }
    cout << "Position loaded" << endl;

    vector<VALUETYPE> boxVectors(9,0);
    if (usePeriodic) {
        Vec3 box[3];
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                boxVectors[i*3+j] = box[i][j]*10;
    } else {
        boxVectors[0] = 99999.9;
        boxVectors[4] = 99999.9;
        boxVectors[8] = 99999.9;
    }
    cout << "Box loaded" << endl;
    
    // run model
    vector<VALUETYPE> force_tmp(positions.size()*3, 0);
    vector<VALUETYPE> virial(9,0);
    ENERGYTYPE ener = 0;
    cout << "pos size:    " << positions.size()  << "    ";
    cout << "virial size: " << virial.size()     << "    ";
    cout << "box size:    " << boxVectors.size() << "    ";
    cout << "types size:  " << types.size()      << "    ";
    cout << "forces size: " << force_tmp.size()  << endl;
    deepmodel.compute(ener, force_tmp, virial, positions, types, boxVectors);
    cout << "Model finished" << endl;

    double energy = 0.0;
    if (includeEnergy) {
        energy = ener*96;
    }
    if (includeForces) {
        // vector<VALUETYPE> data(3*pos.size(),0);
        // for(int i=0;i<mask.size();i++){
        //     int p = mask[i];
        //     for(int j=0;j<3;j++){
        //         data[3*p+j] = force_tmp[3*i+j];
        //     }
        // }
        // networkForces.upload(data);
        networkForces.upload(force_tmp);

        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&networkForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }
    return energy;
}
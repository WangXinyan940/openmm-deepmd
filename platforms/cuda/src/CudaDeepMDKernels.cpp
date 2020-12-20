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
    cu.setAsCurrent();

    int numParticles = system.getNumParticles();
    // hold model
    NNPInter model(force.getFile());
    deepmodel = model;

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
        defines["FORCES_TYPE"] = "float";
        networkForces.initialize(cu, 3*numParticles, sizeof(float), "networkForces");
    #endif
    module = cu.createModule(CudaDeepMDKernelSources::deepMDForce, defines);
    cout << CudaDeepMDKernelSources::deepMDForce << endl;
    addForcesKernel = cu.getKernel(module, "addForces");
    testKernel = cu.getKernel(module, "testForces");

    vector<float> testi;
    vector<float> testj;
    vector<float> testk(50,0);
    for(int i=0;i<50;i++){
        testi.push_back(i+1);
        testj.push_back(3*i-1);
    }
    CudaArray inpi, inpj, inpk;
    inpi.initialize(cu, 50, sizeof(float), "inpi");
    inpj.initialize(cu, 50, sizeof(float), "inpj");
    inpk.initialize(cu, 50, sizeof(float), "inpk");
    inpi.upload(testi);
    inpj.upload(testj);
    inpk.upload(testk);
    void* argtest[] = {&inpi.getDevicePointer(), &inpj.getDevicePointer(), &inpk.getDevicePointer()};
    cout << "before send in initialize" << endl;
    cu.executeKernel(testKernel, argtest, 10);
    inpk.download(testk);
    cout << testk[0] << " " << testk[10] << endl;
}

double CudaCalcDeepMDForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    vector<float> testi;
    vector<float> testj;
    vector<float> testk(50,0);
    for(int i=0;i<50;i++){
        testi.push_back(i+1);
        testj.push_back(3*i-1);
    }
    CudaArray inpi, inpj, inpk;
    inpi.initialize(cu, 50, sizeof(float), "inpi");
    inpj.initialize(cu, 50, sizeof(float), "inpj");
    inpk.initialize(cu, 50, sizeof(float), "inpk");
    inpi.upload(testi);
    inpj.upload(testj);
    inpk.upload(testk);
    void* argtest[] = {&inpi.getDevicePointer(), &inpj.getDevicePointer(), &inpk.getDevicePointer()};
    cout << "before send in execute1" << endl;
    cu.executeKernel(testKernel, argtest, 10);
    inpk.download(testk);
    cout << testk[0] << " " << testk[10] << endl;
    
    vector<Vec3> pos;
    context.getPositions(pos);
    int numParticles = cu.getNumAtoms();

    cout << "Goes in" << endl;
    
    vector<VALUETYPE2> positions(3*mask.size(),0.0);
    for (int i = 0; i < mask.size(); i++) {
        positions[3*i] = pos[mask[i]][0]*10;
        positions[3*i+1] = pos[mask[i]][1]*10;
        positions[3*i+2] = pos[mask[i]][2]*10;
    }
    // cout << "Position loaded" << endl;

    vector<VALUETYPE2> boxVectors(9,0);
    if (usePeriodic) {
        Vec3 box[3];
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                boxVectors[i*3+j] = box[i][j]*10;
    } else {
        boxVectors[0] = 9999.9;
        boxVectors[4] = 9999.9;
        boxVectors[8] = 9999.9;
    }
    // cout << "Box loaded" << endl;
    
    // run model
    vector<VALUETYPE2> force_tmp(positions.size(), 0);
    vector<VALUETYPE2> virial(9,0);
    double ener = 0;
    cout << "Before run" << endl;
    deepmodel.compute(ener, force_tmp, virial, positions, types, boxVectors);
    cout << "Model finished" << endl;

    double energy = 0.0;
    if (includeEnergy) {
        energy = ener*96;
    }
    if (includeForces) {
        cout << "Before upload" << endl;
        vector<VALUETYPE> data(3*pos.size(),0);
        for(int i=0;i<mask.size();i++){
            int p = mask[i];
            for(int j=0;j<3;j++){
                data[3*p+j] = force_tmp[3*i+j];
            }
        }
        networkForces.upload(data);
        cout << "After upload" << endl;
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        cout << numParticles << "   " << paddedNumAtoms << endl;

        vector<float> testi2;
        vector<float> testj2;
        vector<float> testk2(50,0);
        for(int i=0;i<50;i++){
            testi2.push_back(i+1);
            testj2.push_back(3*i-1);
        }
        CudaArray inpi2, inpj2, inpk2;
        inpi2.initialize(cu, 50, sizeof(float), "inpi");
        inpj2.initialize(cu, 50, sizeof(float), "inpj");
        inpk2.initialize(cu, 50, sizeof(float), "inpk");
        inpi2.upload(testi2);
        inpj2.upload(testj2);
        inpk2.upload(testk2);
        void* argtest2[] = {&inpi2.getDevicePointer(), &inpj2.getDevicePointer(), &inpk2.getDevicePointer()};
        cout << "before send in execute2" << endl;
        cu.executeKernel(testKernel, argtest2, 10);
        inpk2.download(testk2);
        cout << testk2[0] << " " << testk2[10] << endl;

        void* args[] = {&networkForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }
    return energy;
}
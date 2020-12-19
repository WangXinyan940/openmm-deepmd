#include "ReferenceDeepMDKernels.h"
#include "DeepMDForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"

#ifdef HIGH_PREC
typedef double VALUETYPE2;
#else
typedef float VALUETYPE2;
#endif

using namespace DeepMDPlugin;
using namespace OpenMM;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->positions);
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->forces);
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

ReferenceCalcDeepMDForceKernel::~ReferenceCalcDeepMDForceKernel() {
}

void ReferenceCalcDeepMDForceKernel::initialize(const System& system, const DeepMDForce& force, NNPInter& model) {
    int numParticles = system.getNumParticles();
    // hold model
    deepmodel = model;
#ifdef HIGH_PREC
    cout << "HIGH PREC" << endl;
#else
    cout << "LOW PREC" << endl;
#endif
    // create input tensors
    mask = force.getMask();
    types = force.getType();
    usePeriodic = force.usesPeriodicBoundaryConditions();
}

double ReferenceCalcDeepMDForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& force = extractForces(context);
    int numParticles = pos.size();

    vector<VALUETYPE2> positions(mask.size()*3,0.0);
    for (int i = 0; i < mask.size(); i++) {
        positions[3*i] = pos[mask[i]][0]*10;
        positions[3*i+1] = pos[mask[i]][1]*10;
        positions[3*i+2] = pos[mask[i]][2]*10;
    }
    vector<VALUETYPE2> boxVectors(9,0.0);
    if (usePeriodic) {
        Vec3* box = extractBoxVectors(context);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                boxVectors[3*i+j] = box[i][j]*10;
    } else {
        boxVectors[0] = 9999.9;
        boxVectors[4] = 9999.9;
        boxVectors[8] = 9999.9;
    }

    // run model
    vector<VALUETYPE2> force_tmp(positions.size(),0);
    vector<VALUETYPE2> virial(9,0);
    double ener = 0;
    cout << "positions:" << endl;
    for(int i=0;i<positions.size();i++){
        cout << positions[i] << ", ";
    }
    cout << endl;
    cout << "types:" << endl;
    for(int i=0;i<types.size();i++){
        cout << types[i] << ", ";
    }
    cout << endl;
    cout << "boxVectors:" << endl;
    for(int i=0;i<boxVectors.size();i++){
        cout << boxVectors[i] << ", ";
    }
    cout << endl;
    deepmodel.compute(ener, force_tmp, virial, positions, types, boxVectors);

    double energy = 0.0;
    if (includeEnergy) {
        energy = ener * 96.0;
    }
    if (includeForces) {
        for (int i = 0; i < mask.size(); i++) {
            int p = mask[i];
            force[p][0] += force_tmp[3*i]*960.0;
            force[p][1] += force_tmp[3*i+1]*960.0;
            force[p][2] += force_tmp[3*i+2]*960.0;
        }
    }
    return energy;
}
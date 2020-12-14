#include "ReferenceDeepMDKernels.h"
#include "DeepMDForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"

#ifdef HIGH_PREC
typedef double VALUETYPE;
typedef double ENERGYTYPE;
#else
typedef float VALUETYPE;
typedef double ENERGYTYPE;
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

void ReferenceCalcDeepMDForceKernel::initialize(const System& system, const DeepMDForce& force, const NNPInter& model) {
    int numParticles = system.getNumParticles();
    // hold model
    NNPInter deepmodel = model;

    // create input tensors
    mask = force.getMask();
    types = force.getType();
    usePeriodic = force.usesPeriodicBoundaryConditions();
}

double ReferenceCalcDeepMDForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& force = extractForces(context);
    int numParticles = pos.size();

    vector<VALUETYPE> positions;
    for (int i = 0; i < mask.size(); i++) {
        positions.push_back(pos[mask[i]][0]*10);
        positions.push_back(pos[mask[i]][1]*10);
        positions.push_back(pos[mask[i]][2]*10);
    }
    if (usePeriodic) {
        Vec3* box = extractBoxVectors(context);
        vector<VALUETYPE> boxVectors;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                boxVectors.push_back(box[i][j]*10);
    } else {
        vector<VALUETYPE> boxVectors(9,0.0);
        boxVectors[0] = 99999.9;
        boxVectors[4] = 99999.9;
        boxVectors[8] = 99999.9;
    }

    // run model
    vector<VALUETYPE> force_tmp(positions.size()*3,0);
    vector<VALUETYPE> virial(9,0);
    ENERGYTYPE ener = 0;
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
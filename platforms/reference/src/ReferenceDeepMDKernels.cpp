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

void ReferenceCalcDeepMDForceKernel::initialize(const System& system, const DeepMDForce& force) {
    int numParticles = system.getNumParticles();
    // hold model
    deepmd::DeepPot model(force.getFile());
    deepmodel = model;
#ifdef HIGH_PREC
    cout << "HIGH PREC" << endl;
#else
    cout << "LOW PREC" << endl;
#endif
    dpscale = force.getScale();
    // create input tensors
    types = force.getType();
    usePeriodic = force.usesPeriodicBoundaryConditions();
}

double ReferenceCalcDeepMDForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& force = extractForces(context);
    Vec3* box = extractBoxVectors(context);
    int numParticles = pos.size();

    vector<VALUETYPE2> positions(numParticles*3,0.0);
    for (int i = 0; i < numParticles; i++) {
        for (int j = 0; j < 3; j++){
            VALUETYPE2 pwrite = pos[i][j];
            while (usePeriodic && (pwrite > box[j][j] || pwrite < 0)){
                if (pwrite > box[j][j]){
                    pwrite -= box[j][j];
                } else if (pwrite < 0){
                    pwrite += box[j][j];
                }
            }
            positions[3*i+j] = pwrite * 10;
        }
    }
    vector<VALUETYPE2> boxVectors(9,0.0);
    if (usePeriodic) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                boxVectors[3*i+j] = box[i][j]*10;
    } else {
        boxVectors[0] = 999.9;
        boxVectors[4] = 999.9;
        boxVectors[8] = 999.9;
    }

    // run model
    vector<VALUETYPE2> force_tmp(positions.size(),0);
    vector<VALUETYPE2> virial(9,0);
    double ener = 0;

    deepmodel.compute(ener, force_tmp, virial, positions, types, boxVectors);
    
    double energy = 0.0;
    if (includeEnergy) {
        energy = ener * 96.0 * dpscale;
    }
    if (includeForces) {
        for (int i = 0; i < numParticles; i++) {
            force[i][0] += force_tmp[ 3*i ]*960.0*dpscale;
            force[i][1] += force_tmp[3*i+1]*960.0*dpscale;
            force[i][2] += force_tmp[3*i+2]*960.0*dpscale;
        }
    }
    return energy;
}
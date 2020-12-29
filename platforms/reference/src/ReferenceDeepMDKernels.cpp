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
    NNPInter model(force.getFile());
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
    // save cutoff of graph
    rcut = deepmodel.cutoff()*0.1;
    cout << "Rcut:" << rcut << endl;
    neighborList = NeighborList();
    ex.resize(numParticles);
}

double ReferenceCalcDeepMDForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& force = extractForces(context);
    Vec3* box = extractBoxVectors(context);
    int numParticles = pos.size();

    vector<VALUETYPE2> positions(mask.size()*3,0.0);
    for (int i = 0; i < mask.size(); i++) {
        for (int j = 0; j < 3; j++){
            VALUETYPE2 pwrite = pos[mask[i]][j];
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
        energy = ener * 96.0;
    }
    if (includeForces) {
        for (int i = 0; i < mask.size(); i++) {
            int p = mask[i];
            force[p][0] += force_tmp[ 3*i ]*960.0;
            force[p][1] += force_tmp[3*i+1]*960.0;
            force[p][2] += force_tmp[3*i+2]*960.0;
        }
    }
    return energy;
}
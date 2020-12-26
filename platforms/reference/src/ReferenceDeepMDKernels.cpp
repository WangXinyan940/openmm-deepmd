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
    rcut = deepmodel.cutoff()*0.5;
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
        positions[3*i] = pos[mask[i]][0]*10;
        positions[3*i+1] = pos[mask[i]][1]*10;
        positions[3*i+2] = pos[mask[i]][2]*10;
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

    if (usePeriodic && (rcut > box[0][0]/2 || rcut > box[1][1]/2 || rcut > box[2][2]/2)) {
        // rcut > 1/2 cell, cannot use OpenMM NeighborList
        deepmodel.compute(ener, force_tmp, virial, positions, types, boxVectors);
    } else {
        // rcut < 1/2 cell or noPBC, generate OpenMM NeighborList
        // get NeighborList from OpenMM
        computeNeighborListVoxelHash(neighborList, numParticles, pos, ex, box, usePeriodic, rcut, 0.0);
        // convert to LammpsNeighborList
        vector<int> ilist_vec(numParticles, 0);
        vector<int> numnei(numParticles, 0);
        vector<vector<int>> firstnei_vec(numParticles);
        for(int i=0;i<numParticles;i++){
            ilist_vec[i] = i;
        }
        for(int i=0;i<neighborList.size();i++){
            int pi = neighborList[i].first;
            int pj = neighborList[i].second;
            numnei[pi] += 1;
            firstnei_vec[pi].push_back(pj);
            numnei[pj] += 1;
            firstnei_vec[pj].push_back(pi);
        }
        int * firstnei_ptr[numParticles];
        for(int i=0;i<numParticles;i++){
            int* temp = &(firstnei_vec[i][0]);
            firstnei_ptr[i] = temp;
        }
        LammpsNeighborList lammpsnei(numParticles, &ilist_vec[0], &numnei[0], firstnei_ptr);
        deepmodel.compute(ener, force_tmp, virial, positions, types, boxVectors, 0, lammpsnei, 0);
    }
    
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
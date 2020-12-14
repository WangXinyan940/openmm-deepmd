#include "ReferenceDeepMDKernels.h"
#include "DeepMDForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"

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
    types = force.getTypes();
    doubleModel = force.useDoublePrecision();
}

double ReferenceCalcDeepMDForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& force = extractForces(context);
    int numParticles = pos.size();

    if (doubleModel) {
        vector<double> positions;
        for (int i = 0; i < mask.size(); i++) {
            positions.push_back(pos[mask[i]][0]);
            positions.push_back(pos[mask[i]][1]);
            positions.push_back(pos[mask[i]][2]);
        }
    }
    else {
        vector<float> positions;
        for (int i = 0; i < mask.size(); i++) {
            positions.push_back(pos[mask[i]][0]);
            positions.push_back(pos[mask[i]][1]);
            positions.push_back(pos[mask[i]][2]);
        }
    }
    if (usePeriodic) {
        Vec3 box[3];
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        if (doubleModel) {
            vector<double> boxVectors;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    boxVectors.push_back(box[i][j]);
        }
        else {
            vector<float> boxVectors;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    boxVectors.push_back(box[i][j]);
        }
    } else {
        if (doubleModel){
            vector<double> boxVectors(9,0.0);
            boxVectors[0] = 9999.9;
            boxVectors[4] = 9999.9;
            boxVectors[8] = 9999.9;
        } else {
            vector<float> boxVectors(9,0.0);
            boxVectors[0] = 9999.9;
            boxVectors[4] = 9999.9;
            boxVectors[8] = 9999.9;
        }
    }

    // run model
    if (doubleModel){
        vector<double> force_tmp(positions.size()*3, 0);
        vector<double> virial(9,0);
        double ener = 0;
        model.compute(ener, force_tmp, virial, positions, types, boxVectors);
    } else {
        vector<float> force_tmp(positions.size()*3,0);
        vector<float> virial(9,0);
        double ener = 0;
        model.compute(ener, force_tmp, virial, positions, types, boxVectors);
    }


    double energy = 0.0;
    if (includeEnergy) {
        energy = ener;
    }
    if (includeForces) {
        for (int i = 0; i < mask.size(); i++) {
            int p = mask[i];
            force[p][0] += force_tmp[3*i];
            force[p][1] += force_tmp[3*i+1];
            force[p][2] += force_tmp[3*i+2];
        }
    }
    return energy;
}
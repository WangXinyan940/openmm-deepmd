#include "CudaDeepMDKernels.h"
#include "CudaDeepMDKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

using namespace DeepMDPlugin;
using namespace OpenMM;
using namespace std;

CudaCalcDeepMDForceKernel::~CudaCalcDeepMDForceKernel() {
    if (positionsTensor != NULL)
        TF_DeleteTensor(positionsTensor);
    if (boxVectorsTensor != NULL)
        TF_DeleteTensor(boxVectorsTensor);
}

void CudaCalcDeepMDForceKernel::initialize(const System& system, const DeepMDForce& force, TF_Session* session, TF_Graph* graph,
            TF_DataType positionsType, TF_DataType boxType, TF_DataType energyType, TF_DataType forcesType) {
    cu.setAsCurrent();
    this->session = session;
    this->graph = graph;
    this->positionsType = positionsType;
    this->boxType = boxType;
    this->energyType = energyType;
    this->forcesType = forcesType;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    int numParticles = system.getNumParticles();

    // Construct input tensors.

    int64_t positionsDims[] = {numParticles, 3};
    positionsTensor = TF_AllocateTensor(positionsType, positionsDims, 2, numParticles*3*TF_DataTypeSize(positionsType));
    if (usePeriodic) {
        int64_t boxVectorsDims[] = {3, 3};
        boxVectorsTensor = TF_AllocateTensor(boxType, boxVectorsDims, 2, 9*TF_DataTypeSize(boxType));
    }

    // Inititalize CUDA objects.

    networkForces.initialize(cu, 3*numParticles, TF_DataTypeSize(forcesType), "networkForces");
    map<string, string> defines;
    if (forcesType == TF_FLOAT)
        defines["FORCES_TYPE"] = "float";
    else
        defines["FORCES_TYPE"] = "double";
    CUmodule module = cu.createModule(CudaDeepMDKernelSources::DeepMDForce, defines);
    addForcesKernel = cu.getKernel(module, "addForces");
}

double CudaCalcDeepMDForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3> pos;
    context.getPositions(pos);
    int numParticles = cu.getNumAtoms();
    if (positionsType == TF_FLOAT) {
        float* positions = reinterpret_cast<float*>(TF_TensorData(positionsTensor));
        for (int i = 0; i < numParticles; i++) {
            positions[3*i] = pos[i][0];
            positions[3*i+1] = pos[i][1];
            positions[3*i+2] = pos[i][2];
        }
    }
    else {
        double* positions = reinterpret_cast<double*>(TF_TensorData(positionsTensor));
        for (int i = 0; i < numParticles; i++) {
            positions[3*i] = pos[i][0];
            positions[3*i+1] = pos[i][1];
            positions[3*i+2] = pos[i][2];
        }
    }
    if (usePeriodic) {
        Vec3 box[3];
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        if (boxType == TF_FLOAT) {
            float* boxVectors = reinterpret_cast<float*>(TF_TensorData(boxVectorsTensor));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    boxVectors[3*i+j] = box[i][j];
        }
        else {
            double* boxVectors = reinterpret_cast<double*>(TF_TensorData(boxVectorsTensor));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    boxVectors[3*i+j] = box[i][j];
        }
    }
    vector<TF_Output> inputs, outputs;
    int forceOutputIndex = 0;
    if (includeEnergy)
        outputs.push_back({TF_GraphOperationByName(graph, "energy"), 0});
    if (includeForces) {
        forceOutputIndex = outputs.size();
        outputs.push_back({TF_GraphOperationByName(graph, "forces"), 0});
    }
    vector<TF_Tensor*> inputTensors, outputTensors(outputs.size());
    inputs.push_back({TF_GraphOperationByName(graph, "positions"), 0});
    inputTensors.push_back(positionsTensor);
    if (usePeriodic) {
        inputs.push_back({TF_GraphOperationByName(graph, "boxvectors"), 0});
        inputTensors.push_back(boxVectorsTensor);
    }
    TF_Status* status = TF_NewStatus();
    TF_SessionRun(session, NULL, &inputs[0], &inputTensors[0], inputs.size(),
                  &outputs[0], &outputTensors[0], outputs.size(),
                  NULL, 0, NULL, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMMException(string("Error running TensorFlow session: ")+TF_Message(status));
    TF_DeleteStatus(status);
    double energy = 0.0;
    if (includeEnergy) {
        if (energyType == TF_FLOAT)
            energy = reinterpret_cast<float*>(TF_TensorData(outputTensors[0]))[0];
        else
            energy = reinterpret_cast<double*>(TF_TensorData(outputTensors[0]))[0];
    }
    if (includeForces) {
        const void* data = TF_TensorData(outputTensors[forceOutputIndex]);
        networkForces.upload(data);
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&networkForces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }
    return energy;
}
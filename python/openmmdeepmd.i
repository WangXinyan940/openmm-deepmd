  
%module openmmdeepmd

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>

%{
#include "DeepMDForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

%feature("shadow") DeepMDPlugin::DeepMDForce::DeepMDForce %{
    def __init__(self, *args):
        this = _openmmnn.new_DeepMDForce(args[0])
        try:
            self.this.append(this)
        except Exception:
            self.this = this
%}

namespace DeepMDPlugin {

class DeepMDForce : public OpenMM::Force {
public:
    DeepMDForce(const std::string file);
    const std::string& getFile() const;
    void setMask(std::vector<int> mask);
    const std::vector<int>& getMask() const;
    void setType(std::vector<int> type);
    const std::vector<int>& getType() const;
    void setUseDoublePrecision(bool usedouble);
    bool useDoublePrecision() const;
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;

    /*
     * Add methods for casting a Force to a DeepMDForce.
    */
    %extend {
        static DeepMDPlugin::DeepMDForce& cast(OpenMM::Force& force) {
            return dynamic_cast<DeepMDPlugin::DeepMDForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<DeepMDPlugin::DeepMDForce*>(&force) != NULL);
        }
    }
};

}
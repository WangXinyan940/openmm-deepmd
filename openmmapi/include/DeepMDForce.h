#ifndef OPENMM_DEEPMDFORCE_H_
#define OPENMM_DEEPMDFORCE_H_

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "deepmd/DeepPot.h"
#include <string>
#include <vector>

namespace DeepMDPlugin {

/**
 * This class implements DeepMD-kit force field. 
 */

class DeepMDForce : public OpenMM::Force {
public:
    /**
     * Create a DeepMDForce.  The network is defined by a TensorFlow graph saved
     * to a binary protocol buffer file.
     *
     * @param filename   the path to the file containing the model
     */
    DeepMDForce(std::string filename, double scale);
    /**
     * Set type vector as the input of DeepMD model.
     */
    void setType(std::vector<int> type);
    /**
     * Get the type vector for DeepMD model.
     */
    const std::vector<int>& getType() const;
    /**
     *  Set the path to the file containing the freezed DeepMD-kit model.
     * 
     * @param filename  the path to the file containing the model
     */
    void setFile(std::string& filename);
    /**
     * Get the path to the file containing the freezed DeepMD-kit model.
     */
    const std::string& getFile() const;
    void setScale(double scale);
    double getScale() const;
    /**
     * Set whether this force makes use of periodic boundary conditions.  If this is set
     * to true, the TensorFlow graph must include a 3x3 tensor called "boxvectors", which
     * is set to the current periodic box vectors.
     * 
     * @param periodic  if use PBC
     */
    void setUsesPeriodicBoundaryConditions(bool periodic);
    /**
     * Get whether this force makes use of periodic boundary conditions.
     */
    bool usesPeriodicBoundaryConditions() const;
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    std::string dpfile;
    bool usePeriodic;
    std::vector<int> innertype;
    double dpscale;
};

} // namespace DeepMDPlugin

#endif /*OPENMM_DEEPMDFORCE_H_*/
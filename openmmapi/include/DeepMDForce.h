#ifndef OPENMM_DEEPMDFORCE_H_
#define OPENMM_DEEPMDFORCE_H_

#include "openmm/Context.h"
#include "openmm/Force.h"
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
    DeepMDForce(std::string filename);
    /**
     * Set mask for DeepMD force field. Only the index in mask vector would be
     * included in DeepMD model
     * 
     * @param mask       the list of index included in DeepMD model
     */
    void setMask(std::vector<int> mask);
    /**
     * Get the mask for DeepMD force field.
     */
    const std::vector<int>& getMask() const;
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
    std::vector<int> innermask;
    std::vector<int> innertype;
};

} // namespace DeepMDPlugin

#endif /*OPENMM_DEEPMDFORCE_H_*/
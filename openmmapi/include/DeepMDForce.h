#ifndef OPENMM_DEEPMDFORCE_H_
#define OPENMM_DEEPMDFORCE_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

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
    DeepMDForce(const std::string& filename);
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
    std::string file;
    bool usePeriodic;
    std::vector<int> innermask;
    std::vector<int> innertype;
};

} // namespace DeepMDPlugin

#endif /*OPENMM_DEEPMDFORCE_H_*/
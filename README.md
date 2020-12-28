OpenMM DeepMD-kit Plugin
========================

This OpenMM plugin enables applying DeepMD-kit force field to simulation, which support
- Reference Platform [OK]
- CUDA Platform [OK]

Details of DeepMD-kit can be found here: https://github.com/deepmodeling/deepmd-kit

Building This Plugin
====================
Users can build this project by CMake with the followed steps:
1. Prepare C++ API of tensorflow and DeepMD-kit. See https://github.com/deepmodeling/deepmd-kit/blob/master/doc/install.md#install-the-c-interface. The precision of DeepMD-kit should be selected in this step by setting CMake variable FLOAT_PREC to "high" or "low".
2. Build OpenMM. See http://docs.openmm.org/latest/userguide/application.html#installing-openmm. 
3. Use CMake to compile this project. You should at least set OPENMM_DIR, DEEPMD_DIR, TENSORFLOW_DIR. If you use double precision deepmd-kit, DEEPMD_HIGH_PREC should be turned on.
4. Use 'make', 'make install' and then 'make PythonInstall' to install the plugin as openmmdeepmd in your python environment.

Python API
==========
TBD

CUDA Kernels
============

Although we have prepared two implements for Reference and CUDA Platform, using CUDA Platform is still not recommended because:
- Rate determining step is DeepMD calculation (~0.1 s/step), not forcefield calculation. 
- Switching CUDA contexts between OpenMM and Tensorflow would cause some unknown bug.

License
=======

This is a plugin for OpenMM molecular simulation toolkit

Portions copyright (c) 2020 the Authors.

Authors: Xinyan Wang

Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.

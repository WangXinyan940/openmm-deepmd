# openmm-deepmd
This OpenMM plugin enables applying DeepMD-kit force field to simulation, which support
- Reference Platform [OK]
- CUDA Platform [OK]

Details of DeepMD-kit can be found here: [https://github.com/deepmodeling/deepmd-kit]

Although we have prepared two implements for Reference and CUDA Platform, using CUDA Platform is still not recommended because:
- Rate determining step is DeepMD calculation (~0.1 s/step), not forcefield calculation. 
- Neighbor list in CUDA Platform cannot be directly used in DeepMD model.


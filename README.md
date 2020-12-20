# openmm-deepmd
OpenMM plugin to use DeepMD-kit force field, which support
- Reference Platform [OK]
- CUDA Platform [OK]

Using CUDA Platform is not recommended because:
- Rate determining step is DeepMD calculation (~0.1 s/step), not forcefield calculation. 
- Neighbor list in CUDA Platform cannot be directly used in DeepMD model.


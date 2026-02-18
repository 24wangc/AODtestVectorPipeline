# Test Vector Pipeline Overview
Current pipeline supports the following workflows: 

ATLAS AOD -> HDF5 -> text Test Vector

ATLAS AOD -> ROOT -> text Test Vector

This repository contains a modular data-processing pipeline for converting ATLAS AOD files into text file test vectors, with a focus on producing standardized, configurable test vectors for fast algorithm development and validation.

## Build and Usage
Requirements:
- ROOT (with xAOD support)
- HDF5 (C++ bindings)
- C++17 compiler

### Setup ATLAS environment:
```
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
alias setupATLAS='source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh'
setupATLAS
asetup AnalysisBase,25.2.29
```

### Configuring and running the C++ HDF5er:
File: AODtoHDF5/AODtoHDF5v2.cc
Choose desired collections and variables (double and unsigned int types) in lines 285 - 307

Set variable definitions in lines 311 - 368

Run in root:
```
root
gSystem->Load("libxAODRootAccess");
xAOD::Init().ignore();
.L AODtoHDF5v2.cc
HDF5er(true, true, 3)
```

This produces intermediate HDF5 files.

### Configuring and running the Python HDF5er:
Directory: AODtoHDF5configurable/

Edit config.yaml with desired collections, variables, and getter functions
Note: all collections *must* have Et as a double_variable for storage of offsets
run in terminal
```
python3 pipelineStep1.py
```

### Running the C++ HDF5toTestVector:
File: HDF5toTestVector/HDF5toTestVector.cc

```
root
.L HDF5toTestVector.cc
HDF5toTestVector(h5filepath)
```
This produces the final test vector files.

### Configuring and running the Python HDF5toTestVector:
Files: HDF5toTestVector/config.yaml and HDF5toTestVector/HDF5toTestVector.py

Edit config.yaml with desired input and output files, variables (bit length, range, packing order), and collections.
Random data options: set random bool = true to generate random data following the variable specifications. Fixed edge case data can also be generated with the "fixed" option under each variable. Setting fixed = 0 will give edge cases where that variable is all zeros, fixed = 1 for all ones, and fixed = 2 for both all zeros and all ones.

```
python3 HDF5toTestVector.py
```

## Note:
This pipeline is actively evolving. Further steps include:
- implementing selection of events in the intermediate file to test vector stage (ex. only events above a threshold Et value)
- supporting a variety of input formats including raw HDF5, HEPMC, LHE, TTree

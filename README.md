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

### Configuring and running the HDF5er:
Choose desired collections and variables (double and unsigned int types) in lines 285 - 307

Set variable definitions in lines 311 - 368

Run in root:
```
root
.L AODtoHDF5v2.cc
HDF5er(true, true, 3)
```

This produces intermediate HDF5 files.

### Configuring and running the Python HDF5er:
Edit config.yaml with desired collections, variables, and getter functions
Note: all collections *must* have Et as a double_variable for storage of offsets
run in terminal
```
python3 AODtoHDF5.py
```

### Running writeTestVectors:
```
root
.L HDF5toTestVector.cc
HDF5toTestVector(h5filepath)
```
This produces the final test vector files.

## Note:
This pipeline is actively evolving. Further steps include:
- implementing selection of events in the intermediate file to test vector stage (ex. only events above a threshold Et value)
- supporting a variety of input formats including raw HDF5, HEPMC, LHE, TTree

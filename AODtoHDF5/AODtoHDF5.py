"""
This is the python version of the AOD to HDF5 conversion script. This was created due to issues
with C++ and Athena, though time tests must be done to evaluate the pros and cons of the languages. 
This is a current functional version 01/22/2026.
Configurable lines are marked with "CONFIGURABLE" in the comments
"""

import numpy as np
import h5py
from dataclasses import dataclass, field
import ROOT
import math
import time

ROOT.xAOD.Init() # Initialize the xAOD infrastructure
ROOT.gSystem.Load("libxAODRootAccess")


# --- CONFIGURABLE ---
# define the getter functions
def get_gfex(obj, v: str) -> float:
    if v == "Et":
        return obj.et() / 1000.0
    if v == "Eta":
        return obj.eta()
    if v == "Phi":
        return obj.phi()
    raise RuntimeError(f"gFEX unknown var: {v}")


def get_hlt_jet(obj, v: str) -> float:
    if v == "Et":
        return obj.e() / (1000.0 * math.cosh(obj.eta()))
    if v == "Eta":
        return obj.eta()
    if v == "Phi":
        return obj.phi()
    raise RuntimeError(f"HLT unknown var: {v}")


def get_reco_ak10(obj, v: str) -> float:
    if v == "Et":
        return obj.e() / (1000.0 * math.cosh(obj.eta()))
    if v == "Eta":
        return obj.eta()
    if v == "Phi":
        return obj.phi()
    if v == "Mass":
        return obj.m() / 1000.0
    raise RuntimeError(f"RecoAK10 unknown var: {v}")

# -- END OF CONFIGURABLE ---

"""
Create the offsets class to deal with jagged arrays
"""
@dataclass
class JaggedOffsets:
    offsets: list[int] = field(default_factory=list)
    
    def endEvent(self, currentSize: int):
        self.offsets.append(currentSize)

""" 
create the Jet Collection class with:
    name: collection name ()
    double_vars: list of kinematic variable names (et, eta, phi)
    int_vars: list of integer variable names (et_index)
    kinematics: a dictionary mapping variable name to list of values
    indices: same as kinematics
    offsets: to store jagged arrays
"""
@dataclass
class JetCollection:
    name: str
    # stores the names of the double and integer variables
    double_vars: list[str] = field(default_factory=lambda: ["Et","Eta","Phi"])
    int_vars: list[str] = field(default_factory=lambda: ["EtIndex"])

    # maps the names of the variables to the actual data
    kinematics: dict[str, list[float]] = field(init=False)
    indices: dict[str, list[int]] = field(init=False)

    # stores the offsets
    offsets: JaggedOffsets = field(default_factory=JaggedOffsets)

    def __post_init__(self):
        self.kinematics = {v: [] for v in self.double_vars}
        self.indices = {v: [] for v in self.int_vars}

''' fill the collection of one event into the corresponding JetCollection
collection is the container retrieved from the actual AOD file itself
take as parameters the collection, the JetCollection, and the functions to retrieve the desired variables
'''
def fillJetCollectionOneEvent(collection, jc:JetCollection, getVar):
    # check if collection exists
    if collection is None:
        jc.offsets.endEvent(len(jc.kinematics["Et"]))
        return

    # loop through the collection
    for i, obj in enumerate(collection):
        # store the index in jc
        jc.indices["EtIndex"].append(i)

        # store data for each kin var
        for v in jc.double_vars:
            # use the getter function to get the kinematic variable data
            jc.kinematics[v].append(float(getVar(obj, v)))

    # mark the end of event in offsets
    jc.offsets.endEvent(len(jc.kinematics["Et"]))

# function for retrieving the collection from the event and filling it to the jetCollection structure
def retrieveAndFill(event, key, jc, getVar, container_type = None):
    collection = getattr(event, key, None)
    fillJetCollectionOneEvent(collection, jc, getVar)

# function for creating the group in the hdf5 file
def create_group(h5, path):
    if (path not in h5):
        h5.create_group(path)

# function to write a data vector as a dataset to the hdf5 file at a given path
def write_vector(h5: h5py.File, path: str, data):
    # i dont know.... TODO
    arr = np.asarray(data)
    h5.create_dataset(path, data = arr)

# function for writing the jet collection to the hdf5 file
def write_jet_collections(h5, jc:JetCollection):
    base = f"/{jc.name}"

    # create the offsets group
    create_group(h5, base)
    create_group(h5, f"{base}/Et")
    write_vector(h5, f"{base}/Et/offsets", jc.offsets.offsets)

    # fill vector data
    for vname in jc.double_vars:
        # create the base path for the variable name
        path = f"{base}/{vname}"
        create_group(h5, path)

        # create the data path for the variable
        data_path = f"{path}/data"
        # if this variable in jc.kinematics has data, write the data
        if (len(jc.kinematics.get(vname)) != 0):
            write_vector(h5, data_path, jc.kinematics.get(vname))
        else:
            empty = []
            write_vector(h5, data_path, empty)

    # fill index data
    for vname in jc.int_vars:
        # create the base path for the variable name
        path = f"{base}/{vname}"
        create_group(h5, path)

        # create the data path for the variable
        data_path = f"{path}/data"
        # if this variable in jc.kinematics has data, write the data
        if (len(jc.indices.get(vname)) != 0):
            write_vector(h5, data_path, jc.indices.get(vname))
        else:
            empty = []
            write_vector(h5, data_path, empty)

def HDF5er(root_filename: str, out_h5: str, max_events):
    '''
    Convert AOD files into intermediate HDF5 format with jagged offsets
    '''
    # start setup timer
    t_setup_start = time.perf_counter()

    # Open ROOT file
    f = ROOT.TFile.Open(root_filename)
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open {root_filename}")

    event = ROOT.xAOD.MakeTransientTree(f, "CollectionTree")

    n_entries = int(event.GetEntries())
    print("Number of events:", n_entries)

    # --- CONFIGURABLE ---
    # JetCollections
    jc_gFEXSRJ = JetCollection("gFEXSRJ", ["Et", "Eta", "Phi"], ["EtIndex"])
    jc_HLTJets = JetCollection("HLTJets", ["Et", "Eta", "Phi"], ["EtIndex"])
    jc_RecoAK10 = JetCollection("RecoAK10", ["Et", "Eta", "Phi", "Mass"], ["EtIndex"])

    jet_collections = [
        (jc_gFEXSRJ, "L1_gFexSRJetRoI", get_gfex),
        (jc_HLTJets, "HLT_AntiKt4EMTopoJets_subjesIS", get_hlt_jet),
        (jc_RecoAK10, "AntiKt10UFOCSSKJets", get_reco_ak10),
    ]
    # --- END CONFIGURABE ---

    # end setup timer
    t_setup_end = time.perf_counter()

    # start event timer
    t_evt_start = time.perf_counter()

    # Event loop
    for i_evt in range(min(n_entries, max_events)):
        event.GetEntry(i_evt)

        for jc, key, getter in jet_collections:
            retrieveAndFill(event, key, jc, getter)

        if i_evt % 100 == 0:
            print(f"{i_evt} events processed")

    f.Close()
    # end timing
    t_evt_end = time.perf_counter()
    evt_total_time = t_evt_end - t_evt_start

    # start writing time
    t_write_start = time.perf_counter()

    # Write HDF5
    with h5py.File(out_h5, "w") as h5:
        for jc, _, _ in jet_collections:
            write_jet_collections(h5, jc)

    print("Wrote:", out_h5)
    t_write_end = time.perf_counter()
    write_total_time = t_write_end - t_write_start

    print("\n[Timing] Event processing")
    print(f"  Events processed: {max_events}")
    print(f"  Total time: {evt_total_time:.3f} s")
    print(f"  Setup time: {t_setup_end - t_setup_start} s")
    print(f"  Avg time/event: {evt_total_time / max_events:.6f} s")
    print(f"  HDF5 writing time: {write_total_time}")


root_file = "/data/crystalwang/testVectorPipeline/testData/DAOD_TrigGepPerf/Signal_HHbbbb_VBF/DAOD_JETM42.SIGNAL_HHBBBB_VBF_DAOD_JETM42.root"
out_h5 = "/data/crystalwang/testVectorPipeline/testData/outputHDF5Files/pythonHDF5.h5"
HDF5er(root_file, out_h5, 100)
/**
This is the current working version of AOD to HDF5. Employes templated functions and requires only changes in lines 281 and 307
to customize collections
12/26/2025
 */

// To execute: e.g., root ; .L nTupler.C ; nTupler(true, true, true) 
#include <algorithm>
#include <numeric>   // std::iota
#include <H5Cpp.h>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <filesystem> // C++17
#include <algorithm>
#include <stdexcept>  // for std::runtime_error
//#include "xAODCutFlow/CutBookkeeperContainer.h"
//#include "xAODCutFlow/CutBookkeeper.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TH2F.h"
#include <functional>
#include <limits>
#include "../analysisHelperFunctions.h"
using std::string;
using std::cout;
using std::cerr;
using std::endl;

// Settings
const bool afBool = true;
const bool vbfBool = true;
const unsigned int nJZSlices = 10;
namespace fs = std::filesystem;

// Constants required for reweighting

// In barns^-1 - 7.5*10^34 cm^-2 s^-1 * 1 s (HL-LHC 200 PU inst. lumi * 1 second) - use 1 second to make rates plots easy
const double reweightLuminosity = 7.5e10;

// Filter efficiencies from AMI [in b]
const double filterEffienciesByJZSlice[nJZSlices] = {0.9716436,    // JZ0
                                                     0.03777559,   // JZ1
                                                     0.01136654,   // JZ2
                                                     0.01367042,   // JZ3
                                                     0.01628158,   // JZ4
                                                     0.01905588,   // JZ5
                                                     0.01352844,   // JZ6
                                                     0.01764909,   // JZ7
                                                     0.01887484,   // JZ8
                                                     0.02827565};  // JZ9

// Cross sections from AMI [in b]
const double crossSectionsByJZSlice[nJZSlices] = {0.07893,      // JZ0
                                                  0.09679,      // JZ1
                                                  0.0026805,    // JZ2
                                                  0.000029984,  // JZ3
                                                  2.972e-7,     // JZ4
                                                  5.5384e-09,   // JZ5
                                                  3.2616e-10,   // JZ6
                                                  2.1734e-11,   // JZ7
                                                  9.2995e-13,   // JZ8
                                                  3.4519e-14};  // JZ9

// From CutBookkeeper printed from python script
const double sumOfEventWeightsByJZSlice[nJZSlices] = {10000.0,                // JZ0
                                                      467.7515499125784,      // JZ1
                                                      3.9288560244923474,     // JZ2
                                                      0.08040490627634789,    // JZ3
                                                      0.0025011268072319126,  // JZ4
                                                      0.00019515390387392362, // JZ5
                                                      3.325120293798389e-05,  // JZ6
                                                      9.518299562145949e-06,  // JZ7 
                                                      2.8526936132339253e-06, // JZ8
                                                      3.274964361399163e-09}; // JZ9 

// Helper to get input DAOD ntuple file names
std::string makeFileDir(bool vbfBool, bool signalBool, int jzSlice) {
    static const std::string kRoot = "/data/crystalwang/testVectorPipeline/testData";
    return kRoot + "/DAOD_TrigGepPerf/Signal_HHbbbb_VBF/";
}

//HDF5 helper functions to convert regular data types into HDF5 native types
template<typename T> H5::PredType h5type();
template<> H5::PredType h5type<double>() { return H5::PredType::NATIVE_DOUBLE; }
template<> H5::PredType h5type<int>()    { return H5::PredType::NATIVE_INT; }
template<> H5::PredType h5type<unsigned int>() { return H5::PredType::NATIVE_UINT; }

//function to write a data vector as a dataset to the hdf5 file at a given path
template<typename T>
void write_vector(H5::H5File& file, const string& path, const std::vector<T>& data) {
    //LEARN
    hsize_t dims[1] = { static_cast<hsize_t>(data.size()) };
    H5::DataSpace space(1, dims);
    H5::DataSet dset = file.createDataSet(path, h5type<T>(), space);
    if (!data.empty())
        dset.write(data.data(), h5type<T>());
}

//jagged array structure to deal with awkward arrays
//LEARN
struct JaggedOffsets {
    std::vector<int> offsets;
    void endEvent(size_t currentSize) {
        offsets.push_back(static_cast<int>(currentSize));
    }
};

// create a structure to store collection information
// name will be the collection, will store map of kinematic variables
struct JetCollection {
    std::string name;

    std::vector<std::string> double_vectors; // things like et values, eta values
    std::vector<std::string> int_vectors; //et index

    bool leading;
    bool subleading;

    // map the variable ("et", "eta") to the actual data vector
    std::unordered_map<std::string, std::vector<double>> kinematics;
    std::unordered_map<std::string, std::vector<unsigned int>> indices;

    std::unordered_map<std::string, std::vector<double>> leading_kinematics;
    std::unordered_map<std::string, std::vector<double>> subleading_kinematics;

    JaggedOffsets offsets;

    // Constructor with default fields (Et, Eta, Phi and EtIndex)
    JetCollection(const std::string& groupName,
                  std::vector<std::string> doubles = {"Et","Eta","Phi"},
                  std::vector<std::string> ints   = {"EtIndex"},
                  bool storeLeading = true,
                  bool storeSubleading = true)
        : name(groupName),
          double_vectors(std::move(doubles)),
          int_vectors(std::move(ints)),
          leading(storeLeading),
          subleading(storeSubleading)
    {
        // allocate the maps based on the names
        for (const auto& n : double_vectors) {
            kinematics[n] = {};
            if (leading) {
                leading_kinematics[n] = {};
            }
            if (subleading) {
                subleading_kinematics[n] = {};
            }
        }

        for (const auto& n : int_vectors) {
            indices[n] = {};
        }
    }
};

// Push the information into the JetCollection for one event
// VarGetter is a function that retrieves the desired bariable value
template <typename ObjT>
using VarGetter = std::function<double(const ObjT* obj, const std::string& var)>;

// fill one event into JetCollection
template<typename Container, typename ObjT>
void fillJetCollectionOneEvent(const Container* collection, JetCollection& jc, VarGetter<ObjT> getVar) {
    // check if collection
    if(!collection) {
        jc.offsets.endEvent(jc.kinematics.at("Et").size());
        return;
    }

    // loop through the collection
    for (size_t i = 0; i < collection->size(); i++) {
        const ObjT* obj = (*collection)[i];
        if (!obj) continue;

        jc.indices["EtIndex"].push_back(static_cast<unsigned int>(i));
        
        // fill requested doubles like Et, Eta, and Phi
        for (const auto& vname : jc.double_vectors) {
            jc.kinematics[vname].push_back(getVar(obj, vname));
        }
    }

    jc.offsets.endEvent(jc.kinematics["Et"].size());
}

// retrieve the collection from the event and fill it to the jetcollection structure
template<typename Container, typename ObjT>
void retrieveAndFill(
    xAOD::TEvent& event,
    const std::string& key,
    JetCollection& jc,
    VarGetter<ObjT> getVar
) {
    // retrieve colletion
    const Container* cont = nullptr;
    if (!event.retrieve(cont, key).isSuccess()) {
        cont = nullptr; // treat as missing
    }

    // fill jetcollection
    fillJetCollectionOneEvent<Container, ObjT>(cont, jc, getVar);
}

// Recursively find first non-Higgs daughters
void find_non_higgs_daughters(const xAOD::TruthParticle* particle,
                              std::vector<const xAOD::TruthParticle*>& result) {
    if (!particle) return;

    // If the particle is NOT a Higgs, store it directly
    if (particle->pdgId() != 25) {
        result.push_back(particle);
        return;
    }

    // If it is a Higgs, recurse through its children
    for (unsigned int i = 0; i < particle->nChildren(); ++i) {
        find_non_higgs_daughters(particle->child(i), result);
    }
}


// create groups and write jet collection to h5 file
void writeJetCollection(H5::H5File& h5, const JetCollection& jc) {
    const std::string base = "/" + jc.name;

    // create the base group for the collection
    auto safe_create = [&](const std::string& path) {
        try {
            h5.createGroup(path);
        } catch (H5::Exception&) {
            // already exists → ignore
        }
    };

    // create the offsets
    safe_create(base);
    safe_create(base + "/Et");
    write_vector(h5, base + "/Et/offsets", jc.offsets.offsets);

    //fill vector data
    for (const auto& vname : jc.double_vectors) {
        const std::string path = base + "/" + vname;
        safe_create(path);

        std::string data_path = path + "/data";
        auto it = jc.kinematics.find(vname);
        if (it != jc.kinematics.end()) {
            write_vector(h5, data_path, it->second);
        } else {
            std::vector<double> empty;
            write_vector(h5, data_path, empty);
        }
    }

    // fill integer data
    for (const auto& vname : jc.int_vectors) {
        const std::string path = base + "/" + vname;
        safe_create(path);

        std::string data_path = path + "/data";
        auto it = jc.indices.find(vname);
        if (it != jc.indices.end()) {
            write_vector(h5, data_path, it->second);
        } else {
            std::vector<unsigned int> empty;
            write_vector(h5, data_path, empty);
        }
    }
    
}

void HDF5er(bool vbfBool, bool signalBool, int jzSlice) {
    // Setup file paths based on whether processing signal or background, and vbf production or ggF production
    std::string fileDir = makeFileDir(vbfBool, signalBool, jzSlice);
    std::cout << "fileDir: " << fileDir << "\n";

    // make the jet collections
    // TODO: get this info from a config file
    JetCollection jc_gFEXSRJ("gFEXSRJ", {"Et","Eta","Phi"}, {"EtIndex"}, false, false);
    JetCollection jc_gFEXLRJ("gFEXLRJ", {"Et","Eta","Phi"}, {"EtIndex"}, false, false);

    JetCollection jc_jFEXSRJ("jFEXSRJ", {"Et","Eta","Phi"}, {"EtIndex"}, false, false);
    JetCollection jc_jFEXLRJ("jFEXLRJ", {"Et","Eta","Phi"}, {"EtIndex"}, false, false);

    JetCollection jc_HLTJets("HLTJets", {"Et","Eta","Phi"}, {"EtIndex"}, false, false);

    JetCollection jc_RecoAK10("RecoAK10", {"Et","Eta","Phi","Mass"}, {"EtIndex"}, false, false);

    JetCollection jc_InTimeAntiKt4TruthJets("InTimeAntiKt4TruthJets", {"Et","Eta","Phi"}, {"EtIndex"}, false, false);

    JetCollection jc_AntiKt4TruthDressedWZJets("AntiKt4TruthDressedWZJets", {"Et","Eta","Phi","Mass"}, {"EtIndex"}, false, false);

    // For writing in a loop later, get a list of the jet collections
    std::vector<JetCollection*> jetCollections = {
        &jc_gFEXSRJ, &jc_gFEXLRJ,
        &jc_jFEXSRJ, &jc_jFEXLRJ,
        &jc_HLTJets,
        &jc_RecoAK10,
        &jc_InTimeAntiKt4TruthJets,
        &jc_AntiKt4TruthDressedWZJets
    };

    // define how to retrieve all the variables for each collection
    // TODO: make this a registry or config file thing
    auto get_gfex = [](const xAOD::gFexJetRoI* j, const std::string& v)->double{
        if (v == "Et")  return j->et() / 1000.0;
        if (v == "Eta") return j->eta();
        if (v == "Phi") return j->phi();
        throw std::runtime_error("gFEX unknown var: " + v);
    };

    auto get_jfex_sr = [](const xAOD::jFexSRJetRoI_v1* j, const std::string& v)->double{
        if (v == "Et")  return j->et() / 1000.0;
        if (v == "Eta") return j->eta();
        if (v == "Phi") return j->phi();
        throw std::runtime_error("jFEX SR unknown var: " + v);
    };

    auto get_jfex_lr = [](const xAOD::jFexLRJetRoI_v1* j, const std::string& v)->double{
        if (v == "Et")  return j->et() / 1000.0;
        if (v == "Eta") return j->eta();
        if (v == "Phi") return j->phi();
        throw std::runtime_error("jFEX LR unknown var: " + v);
    };

    auto get_hlt = [](const xAOD::Jet* j, const std::string& v)->double{
        if (v == "Et")  return j->e() / (1000.0 * std::cosh(j->eta()));
        if (v == "Eta") return j->eta();
        if (v == "Phi") return j->phi();
        throw std::runtime_error("HLT unknown var: " + v);
    };

    auto get_reco = [](const xAOD::Jet* j, const std::string& v)->double{
        if (v == "Et")   return j->e() / (1000.0 * std::cosh(j->eta()));
        if (v == "Eta")  return j->eta();
        if (v == "Phi")  return j->phi();
        if (v == "Mass") return j->m() / 1000.0;
        throw std::runtime_error("RecoAK10 unknown var: " + v);
    };

    auto get_truth_noMass = [](const xAOD::Jet* j, const std::string& v)->double{
        if (v == "Et")  return j->e() / (1000.0 * std::cosh(j->eta()));
        if (v == "Eta") return j->eta();
        if (v == "Phi") return j->phi();
        throw std::runtime_error("Truth unknown var: " + v);
    };

    auto get_truth_withMass = [](const xAOD::Jet* j, const std::string& v)->double{
        if (v == "Et")   return j->e() / (1000.0 * std::cosh(j->eta()));
        if (v == "Eta")  return j->eta();
        if (v == "Phi")  return j->phi();
        if (v == "Mass") return j->m() / 1000.0;
        throw std::runtime_error("TruthWZ unknown var: " + v);
    };

    auto has_prefix = [](const std::string& s, const std::string& p) {
        return s.rfind(p, 0) == 0; // starts with
    };
    auto has_suffix = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() &&
            s.compare(s.size() - suf.size(), suf.size(), suf) == 0; // ends with
    };

    // Loop 1: DAOD_JETM42*.root (from derivation of an AOD)
    std::vector<std::string> fileNames;
    for (const auto& entry : fs::directory_iterator(fileDir)) {
        if (!entry.is_regular_file()) continue;
        const std::string fn = entry.path().filename().string();
        if (has_prefix(fn, "DAOD_JETM42") && has_suffix(fn, ".root")) {
            fileNames.push_back(entry.path().string());
        }
    }

    // give the error if no files were found
    if (fileNames.empty()) {
    std::cout << "No ROOT files found in directory." << endl;
    return;
    }

    //print the number of files found
    std::cout << "Found " << fileNames.size() << " files." << endl;

    // Main processing loop
    // loop through file names
    int fileIt = 0;
    unsigned int higgsPassEventCounter = 0;

    for (const auto& fileName : fileNames) {
        fileIt++; 
        //if (fileIt > 9) break; 
        cout << "Processing file: " << fileName << endl;

        TFile* f = TFile::Open(fileName.c_str());
        if (!f || f->IsZombie()) {
            cerr << "Could not open " << fileName << endl;
            continue;
        }

        // Open the DAOD file
        xAOD::TEvent event(xAOD::TEvent::kClassAccess);
        if (!event.readFrom(f).isSuccess()) {
            cerr << "Cannot read xAOD from file." << endl;
            continue;
        }

        std::cout << "  Number of events: " << event.getEntries() << endl;

        unsigned int passedEventsCounter = 0;
        unsigned int skippedEventsEmptyTruth = 0;
        unsigned int skippedEventsHSTP = 0;

        for (Long64_t iEvt = 0; iEvt < event.getEntries(); ++iEvt) { // NOTE assume that # of events for GEP and DAOD files is the same, else will get a seg fault.
            //std::cout << "iEvt: " << iEvt << "\n";
            // Retrieve truth and in time pileup jets first to apply hard-scatter-softer-than-PU (HSTP) filter (described here: https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/JetEtMissMCSamples#Dijet_normalization_procedure_HS)
            // Also require that truth jet collection is not empty
            if (iEvt > 100) {
                break;
            }

            event.getEntry(iEvt);               // DAOD event iEvt

            // L1Calo jets vectors
            retrieveAndFill<xAOD::gFexJetRoIContainer, xAOD::gFexJetRoI>(
                event, "L1_gFexSRJetRoI", jc_gFEXSRJ, get_gfex);

            retrieveAndFill<xAOD::gFexJetRoIContainer, xAOD::gFexJetRoI>(
                event, "L1_gFexLRJetRoI", jc_gFEXLRJ, get_gfex);

            retrieveAndFill<DataVector<xAOD::jFexSRJetRoI_v1>, xAOD::jFexSRJetRoI_v1>(
                event, "L1_jFexSRJetRoI", jc_jFEXSRJ, get_jfex_sr);

            retrieveAndFill<DataVector<xAOD::jFexLRJetRoI_v1>, xAOD::jFexLRJetRoI_v1>(
                event, "L1_jFexLRJetRoI", jc_jFEXLRJ, get_jfex_lr);

            // HLT jets
            retrieveAndFill<xAOD::JetContainer, xAOD::Jet>(
                event, "HLT_AntiKt4EMTopoJets_subjesIS", jc_HLTJets, get_hlt);

            // In-time truth jets
            retrieveAndFill<xAOD::JetContainer, xAOD::Jet>(
                event, "InTimeAntiKt4TruthJets", jc_InTimeAntiKt4TruthJets, get_truth_noMass);

            // Reco AK10 jets
            retrieveAndFill<xAOD::JetContainer, xAOD::Jet>(
                event, "AntiKt10UFOCSSKJets", jc_RecoAK10, get_reco);

            // Truth dressed WZ jets
            retrieveAndFill<xAOD::JetContainer, xAOD::Jet>(
                event, "AntiKt4TruthDressedWZJets", jc_AntiKt4TruthDressedWZJets, get_truth_withMass);

        } // loop through events
        std::cout << "for jz: " << jzSlice << " these many events passed: " << passedEventsCounter << " out of: " << event.getEntries() << "\n";
        std::cout << " these many events skipped due to empty truth container: " << skippedEventsEmptyTruth << " these many events skipped due to pt hard < pt pileup: " << skippedEventsHSTP << "\n";
        std::cout << "closing files" << "\n";
        f->Close();
    } // loop through filenames

    // get the H5 file
    H5::H5File h5("/data/crystalwang/testVectorPipeline/testData/outputHDF5Files/exampleHDF5.h5", H5F_ACC_TRUNC);

    // Write all jet collections
    for(const JetCollection* jc : jetCollections) {
        writeJetCollection(h5, *jc);
    }
}

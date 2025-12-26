/**
This is the old version of AOD to HDF5. Things are all hardcoded and have since been fixed
12/26/2025
 */
// To execute: e.g., root ; .L nTupler.C ; nTupler(true, true, true) 
#include <algorithm>
#include <numeric>   // std::iota
#include <H5Cpp.h>

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
#include <chrono>
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

//function to write a data vector as a dataset at a given path
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

// helper to make a group in the hdf5
void make_group(H5::H5File& h5, const std::string& collection) {
    auto safe_create = [&](const std::string& path) {
        try {
            h5.createGroup(path);
        } catch (H5::Exception&) {
            // already exists → ignore
        }
    };

    // ensure parent exists
    safe_create("/" + collection);

    // ensure children exist
    safe_create("/" + collection + "/Et");
    safe_create("/" + collection + "/EtIndex");
    safe_create("/" + collection + "/Eta");
    safe_create("/" + collection + "/Phi");
}

void HDF5er(bool vbfBool, bool signalBool, int jzSlice) {
    // Setup file paths based on whether processing signal or background, and vbf production or ggF production
    std::string fileDir = makeFileDir(vbfBool, signalBool, jzSlice);
    std::cout << "fileDir: " << fileDir << "\n";

    //make all the vectors!

    // Event info vectors
    std::vector<double> mcEventWeights;
    double sumOfWeightsForSample;
    int sampleJZSlice;
    std::vector<double> eventWeights;

    // Truth Particle vectors
    //std::vector<int> truthParticlePDGId, truthParticleStatus; 
    //std::vector<double> truthParticleEtValues, truthParticleEnergyValues, truthParticlepTValues, truthParticlepxValues, truthParticlepyValues, truthParticlepzValues, truthParticleEtaValues, truthParticlePhiValues;
    std::vector<unsigned int> higgsIndexValues, indexOfHiggsValues;
    std::vector<double> truthbquarksEtValues, truthbquarksEnergyValues, truthbquarkspTValues, truthbquarkspxValues, truthbquarkspyValues, truthbquarkspzValues, truthbquarksEtaValues, truthbquarksPhiValues;
    std::vector<double> truthHiggsEtValues, truthHiggsEnergyValues, truthHiggspTValues, truthHiggspxValues, truthHiggspyValues, truthHiggspzValues, truthHiggsEtaValues, truthHiggsPhiValues, truthHiggsInvMassValues;
    //std::vector<double> truthVBFQuarkValues, truthVBFQuarkEnergyValues, truthVBFQuarkpTValues, truthVBFQuarkpxValues, truthVBFQuarkpyValues, truthVBFQuarkpzValues, truthVBFQuarkEtaValues, truthVBFQuarkPhiValues;
    JaggedOffsets truthHiggsOff;
    JaggedOffsets truthBOff;

    // Tower / cluster vectors
    std::vector<double> caloTopoTowerEtValues, caloTopoTowerEtaValues, caloTopoTowerPhiValues;
    std::vector<double> topo422EtValues, topo422EtaValues, topo422PhiValues;
    JaggedOffsets caloTapoTowerOff;
    JaggedOffsets topo422Off;

    // L1Calo jets vectors
    std::vector<unsigned int> gFexSRJEtIndexValues; // stores index in sorted by Et list of jets
    std::vector<double> gFexSRJEtValues, gFexSRJEtaValues, gFexSRJPhiValues;
    std::vector<double> gFexSRJLeadingEtValues, gFexSRJLeadingEtaValues, gFexSRJLeadingPhiValues;
    std::vector<double> gFexSRJSubleadingEtValues, gFexSRJSubleadingEtaValues, gFexSRJSubleadingPhiValues;
    std::vector<unsigned int> gFexLRJEtIndexValues; // stores index in sorted by Et list of jets
    std::vector<double> gFexLRJEtValues, gFexLRJEtaValues, gFexLRJPhiValues;
    std::vector<double> gFexLRJLeadingEtValues, gFexLRJLeadingEtaValues, gFexLRJLeadingPhiValues;
    std::vector<double> gFexLRJSubleadingEtValues, gFexLRJSubleadingEtaValues, gFexLRJSubleadingPhiValues;
    JaggedOffsets gFexSRJOff;
    JaggedOffsets gFexLRJOff;

    std::vector<unsigned int> jFexSRJEtIndexValues;
    std::vector<double> jFexSRJEtValues, jFexSRJEtaValues, jFexSRJPhiValues;
    std::vector<double> jFexSRJLeadingEtValues, jFexSRJLeadingEtaValues, jFexSRJLeadingPhiValues;
    std::vector<double> jFexSRJSubleadingEtValues, jFexSRJSubleadingEtaValues, jFexSRJSubleadingPhiValues;
    std::vector<unsigned int> jFexLRJEtIndexValues;
    std::vector<double> jFexLRJEtValues, jFexLRJEtaValues, jFexLRJPhiValues;
    std::vector<double> jFexLRJLeadingEtValues, jFexLRJLeadingEtaValues, jFexLRJLeadingPhiValues;
    std::vector<double> jFexLRJSubleadingEtValues, jFexLRJSubleadingEtaValues, jFexLRJSubleadingPhiValues;
    JaggedOffsets jFexLRJOff;
    JaggedOffsets jFexSRJOff;
    
    //std::vector<double> gFexRhoRoIEtValues, gFexRhoRoIEtaValues, gFexRhoRoIPhiValues; // skip adding these for now.

    // HLT jets vectors
    std::vector<unsigned int> hltAntiKt4SRJEtIndexValues; // stores index in sorted by Et list of jets
    std::vector<double> hltAntiKt4SRJEtValues, hltAntiKt4SRJEtaValues, hltAntiKt4SRJPhiValues;
    std::vector<double> hltAntiKt4SRJLeadingEtValues, hltAntiKt4SRJLeadingEtaValues, hltAntiKt4SRJLeadingPhiValues;
    std::vector<double> hltAntiKt4SRJSubleadingEtValues, hltAntiKt4SRJSubleadingEtaValues, hltAntiKt4SRJSubleadingPhiValues;
    JaggedOffsets hltAntiKt4SRJOff;

    // Reco offline jets vectors
    std::vector<unsigned int> recoAntiKt10LRJEtIndexValues;
    std::vector<double> recoAntiKt10LRJEtValues, recoAntiKt10LRJEtaValues, recoAntiKt10LRJPhiValues, recoAntiKt10LRJMassValues;
    std::vector<double> recoAntiKt10LRJLeadingEtValues, recoAntiKt10LRJLeadingEtaValues, recoAntiKt10LRJLeadingPhiValues, recoAntiKt10LRJLeadingMassValues;
    std::vector<double> recoAntiKt10LRJSubleadingEtValues, recoAntiKt10LRJSubleadingEtaValues, recoAntiKt10LRJSubleadingPhiValues, recoAntiKt10LRJSubleadingMassValues;
    JaggedOffsets recoAntiKt10LRJOff;

    // Truth WZ Antikt4 jets vectors
    std::vector<unsigned int> truthAntiKt4WZSRJEtIndexValues;
    std::vector<double> truthAntiKt4WZSRJEtValues, truthAntiKt4WZSRJEtaValues, truthAntiKt4WZSRJPhiValues, truthAntiKt4WZSRJMassValues;
    std::vector<double> truthAntiKt4WZSRJLeadingEtValues, truthAntiKt4WZSRJLeadingEtaValues, truthAntiKt4WZSRJLeadingPhiValues, truthAntiKt4WZSRJLeadingMassValues;
    std::vector<double> truthAntiKt4WZSRJSubleadingEtValues, truthAntiKt4WZSRJSubleadingEtaValues, truthAntiKt4WZSRJSubleadingPhiValues, truthAntiKt4WZSRJSubleadingMassValues;
    JaggedOffsets truthAntiKt4Off;

    // In time anti-kt 4 truth jets vectors
    std::vector<unsigned int> inTimeAntiKt4TruthSRJEtIndexValues;
    std::vector<double> inTimeAntiKt4TruthSRJEtValues, inTimeAntiKt4TruthSRJEtaValues, inTimeAntiKt4TruthSRJPhiValues;
    std::vector<double> inTimeAntiKt4TruthSRJLeadingEtValues, inTimeAntiKt4TruthSRJLeadingEtaValues, inTimeAntiKt4TruthSRJLeadingPhiValues;
    std::vector<double> inTimeAntiKt4TruthSRJSubleadingEtValues, inTimeAntiKt4TruthSRJSubleadingEtaValues, inTimeAntiKt4TruthSRJSubleadingPhiValues;
    JaggedOffsets inTimeAntiKt4TruthSRJOff;

    auto has_prefix = [](const std::string& s, const std::string& p) {
        return s.rfind(p, 0) == 0; // starts with
    };
    auto has_suffix = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() &&
            s.compare(s.size() - suf.size(), suf.size(), suf) == 0; // ends with
    };

    // --- Timing: event processing ---
    auto tStartProcessing = std::chrono::steady_clock::now();
    std::size_t totalEventsLooped = 0;   // total events we iterated over (across all files)

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
            if (iEvt >= 1000) {
                break;
            }
            // time keeping
            totalEventsLooped++;

            event.getEntry(iEvt);               // DAOD event iEvt

            const xAOD::JetContainer* AntiKt4TruthDressedWZJets = nullptr;
            if (!event.retrieve(AntiKt4TruthDressedWZJets, "AntiKt4TruthDressedWZJets").isSuccess()) {
                std::cout << "Failed to retrieve reco Antik4 Truth Dressed WZ jets" << endl;
                skippedEventsEmptyTruth++;
                continue;
            }

            if ( AntiKt4TruthDressedWZJets->size() == 0 ){
                skippedEventsEmptyTruth++;
                continue;
            } 

            const xAOD::JetContainer* InTimeAntiKt4TruthJets = nullptr;
            if (!event.retrieve(InTimeAntiKt4TruthJets, "InTimeAntiKt4TruthJets").isSuccess()) {
                std::cout << "Failed to retrieve Truth jets" << endl;
                continue;
            }

            // Filter out events where hard scatter is softer than PU
            if(AntiKt4TruthDressedWZJets->at(0)->pt() <= InTimeAntiKt4TruthJets->at(0)->pt()){
                skippedEventsHSTP++;
                continue; 
            } 

            passedEventsCounter++;

            if ((iEvt % 10) == 0) std::cout << "iEvt: " << iEvt << "\n";

                        // -- retrieve collections from DAOD ---
            const xAOD::EventInfo_v1* EventInfo = nullptr;
            if (!event.retrieve(EventInfo, "EventInfo").isSuccess()) {
                cerr << "Cannot access EventInfo" << endl;
                continue;
            }
            
            const xAOD::TruthParticleContainer* TruthBosonsWithDecayParticles = nullptr;
            if (!event.retrieve(TruthBosonsWithDecayParticles, "TruthBosonsWithDecayParticles").isSuccess()) {
                cerr << "Cannot access TruthBosonsWithDecayParticles" << endl;
                continue;
            }

            const xAOD::JetContainer* HLT_AntiKt4EMTopoJets_subjesIS = nullptr;
            if (!event.retrieve(HLT_AntiKt4EMTopoJets_subjesIS, "HLT_AntiKt4EMTopoJets_subjesIS").isSuccess()) {
                cerr << "Failed to retrieve HLT jets" << endl;
                continue;
            }

            const xAOD::gFexJetRoIContainer* L1_gFexSRJetRoI = nullptr;
            if (!event.retrieve(L1_gFexSRJetRoI, "L1_gFexSRJetRoI").isSuccess()) {
                std::cerr << "Failed to retrieve gFex SR jets" << std::endl;
                continue;
            }

            const xAOD::gFexJetRoIContainer* L1_gFexLRJetRoI = nullptr;
            if (!event.retrieve(L1_gFexLRJetRoI, "L1_gFexLRJetRoI").isSuccess()) {
                cerr << "Failed to retrieve gFex LR jets" << endl;
                continue;
            }

            const DataVector<xAOD::jFexSRJetRoI_v1>* L1_jFexSRJetRoI = nullptr;
            if (!event.retrieve(L1_jFexSRJetRoI, "L1_jFexSRJetRoI").isSuccess()) {
                std::cerr << "Failed to retrieve jFex SR jets" << std::endl;
                continue;
            }

            const DataVector<xAOD::jFexLRJetRoI_v1>* L1_jFexLRJetRoI = nullptr;
            if (!event.retrieve(L1_jFexLRJetRoI, "L1_jFexLRJetRoI").isSuccess()) {
                std::cerr << "Failed to retrieve jFex SR jets" << std::endl;
                continue;
            }

            /*
            const xAOD::gFexJetRoIContainer* L1_gFexRhoRoI = nullptr;
            if (!event.retrieve(L1_gFexRhoRoI, "L1_gFexRhoRoI").isSuccess()) {
                std::cerr << "Failed to retrieve gFex energy density" << std::endl;
                continue;
            }*/

            const xAOD::JetContainer* AntiKt10UFOCSSKJets = nullptr;
            if (!event.retrieve(AntiKt10UFOCSSKJets, "AntiKt10UFOCSSKJets").isSuccess()) {
                cerr << "Failed to retrieve reco Antik10 UFOCSSK jets" << endl;
                continue;
            }

            // Retrieve the CaloTopoClusters422 container
            const DataVector<xAOD::CaloCluster_v1>* CaloTopoClusters422 = nullptr;
            if (!event.retrieve(CaloTopoClusters422, "CaloTopoClusters422").isSuccess()) {
                std::cerr << "Failed to retrieve CaloTopoClusters422" << std::endl;
                continue;
            }

            // Retrieve the CaloCalAllTopoTowers container
            const DataVector<xAOD::CaloCluster_v1>* CaloCalAllTopoTowers = nullptr;
            if (!event.retrieve(CaloCalAllTopoTowers, "CaloCalAllTopoTowers").isSuccess()) {
                std::cerr << "Failed to retrieve CaloCalAllTopoTowers" << std::endl;
                continue;
            }
 
            sumOfWeightsForSample = 0;
            sampleJZSlice = 0;
            
            // Get sum of weights for sample, individual Monte Carlo event weight, and compute event weight used later for reweighting histograms
            sumOfWeightsForSample = sumOfEventWeightsByJZSlice[jzSlice];
            
            if (signalBool) sampleJZSlice = -1; 
            float mcEventWeight = EventInfo->mcEventWeight();
            //std::cout << "iEvt: " << iEvt << " and event weight: " << eventWeight << "\n";
            mcEventWeights.push_back(mcEventWeight);

                        // Compute weight for histograms 
            double eventWeight = mcEventWeight * crossSectionsByJZSlice[jzSlice] * filterEffienciesByJZSlice[jzSlice] * reweightLuminosity / (sumOfEventWeightsByJZSlice[jzSlice]);
            eventWeights.push_back(eventWeight);

            // Initialize per-event counters
            unsigned int higgs_counter = -1;
            bool higgsPtCutPassed[2] = {false, false}; 
            // --- Loop over TruthParticles (for Higgs and B's) ---
            std::vector<std::vector<float > > allb_list;
            for (const auto& el : *TruthBosonsWithDecayParticles) {
                // Check to see if Higgs
                if (el->pdgId() == 25 && el->status() == 22) {
                    higgs_counter++;

                    // convert from MeV to GeV
                    float pt = el->pt() / 1000.0;
                    float eta = el->eta();
                    float phi = el->phi();
                    float px = el->px() / 1000.0;
                    float py = el->py() / 1000.0;
                    float pz = el->pz() / 1000.0;
                    float energy = el->e() / 1000.0;
                    float et = energy / cosh(eta);

                    // Fill truthHiggsTree variables
                    truthHiggsEtValues.push_back(et);
                    truthHiggsEtaValues.push_back(eta);
                    truthHiggsPhiValues.push_back(phi);
                    truthHiggspTValues.push_back(pt);
                    truthHiggspxValues.push_back(px);
                    truthHiggspyValues.push_back(py);
                    truthHiggspzValues.push_back(pz);
                    truthHiggsEnergyValues.push_back(energy);
                    indexOfHiggsValues.push_back(higgs_counter);

                    if (higgs_counter == 0) { // fill b's for both higgs, and store index of which higgs they correspond to.  
                        if (pt > 0) higgsPtCutPassed[0] = true;
                        std::vector<const xAOD::TruthParticle*> b1_list;
                        find_non_higgs_daughters(el, b1_list);

                        if (b1_list.size() == 2) {
                            TLorentzVector b1, b2;
                            b1.SetPxPyPzE(b1_list[0]->px() / 1000.0, b1_list[0]->py() / 1000.0,
                                        b1_list[0]->pz() / 1000.0, b1_list[0]->e() / 1000.0);
                            b2.SetPxPyPzE(b1_list[1]->px() / 1000.0, b1_list[1]->py() / 1000.0,
                                        b1_list[1]->pz() / 1000.0, b1_list[1]->e() / 1000.0);
                            double mH = (b1 + b2).M();  // invariant mass in GeV
                            truthHiggsInvMassValues.push_back(mH);
                        } // compute invariant mass

                        for (const auto* b : b1_list) {
                            if (!b) continue;

                            float pt   = b->pt() / 1000.0;
                            float eta  = b->eta();
                            float phi  = b->phi();
                            float px   = b->px() / 1000.0;
                            float py   = b->py() / 1000.0;
                            float pz   = b->pz() / 1000.0;
                            float E    = b->e()  / 1000.0;
                            float Et   = E / std::cosh(eta);

                            higgsIndexValues.push_back(0);  // from 1st Higgs
                            truthbquarksEtValues.push_back(Et);
                            truthbquarksEtaValues.push_back(eta);
                            truthbquarksPhiValues.push_back(phi);
                            truthbquarkspTValues.push_back(pt);
                            truthbquarkspxValues.push_back(px);
                            truthbquarkspyValues.push_back(py);
                            truthbquarkspzValues.push_back(pz);
                            truthbquarksEnergyValues.push_back(E);
                        }
                    }

                    if (higgs_counter == 1) {
                        if (pt > 0) higgsPtCutPassed[1] = true;
                        std::vector<const xAOD::TruthParticle*> b2_list;
                        find_non_higgs_daughters(el, b2_list);

                        if (b2_list.size() == 2) {
                            TLorentzVector b1, b2;
                            b1.SetPxPyPzE(b2_list[0]->px() / 1000.0, b2_list[0]->py() / 1000.0,
                                        b2_list[0]->pz() / 1000.0, b2_list[0]->e() / 1000.0);
                            b2.SetPxPyPzE(b2_list[1]->px() / 1000.0, b2_list[1]->py() / 1000.0,
                                        b2_list[1]->pz() / 1000.0, b2_list[1]->e() / 1000.0);
                            double mH = (b1 + b2).M();  // invariant mass in GeV
                            truthHiggsInvMassValues.push_back(mH);
                        }

                        for (const auto* b : b2_list) {
                            if (!b) continue;

                            float pt   = b->pt() / 1000.0;
                            float eta  = b->eta();
                            float phi  = b->phi();
                            float px   = b->px() / 1000.0;
                            float py   = b->py() / 1000.0;
                            float pz   = b->pz() / 1000.0;
                            float E    = b->e()  / 1000.0;
                            float Et   = E / std::cosh(eta);

                            higgsIndexValues.push_back(1);  // from 2nd Higgs
                            truthbquarksEtValues.push_back(Et);
                            truthbquarksEtaValues.push_back(eta);
                            truthbquarksPhiValues.push_back(phi);
                            truthbquarkspTValues.push_back(pt);
                            truthbquarkspxValues.push_back(px);
                            truthbquarkspyValues.push_back(py);
                            truthbquarkspzValues.push_back(pz);
                            truthbquarksEnergyValues.push_back(E);
                        } // loop through b2 list
                    } // if 2nd higgs in event
                } // if higgs truth particle of interest
            } // loop through truth bosons with decay particles
            bool higgsPtCutsPassed = higgsPtCutPassed[0] || higgsPtCutPassed[1];
            if (higgsPtCutsPassed) higgsPassEventCounter++;
            
            // mark event boundary
            truthHiggsOff.endEvent(truthHiggsEtValues.size());
            truthBOff.endEvent(truthbquarksEtValues.size());

            // Loop over clusters and fill Et, Eta, Phi
            unsigned int topotower_it = 0;
            for (const auto* cluster : *CaloCalAllTopoTowers) {
                if (!cluster) continue;

                double et = cluster->e() / cosh(cluster->eta()) / 1000.0; // Et in GeV
                caloTopoTowerEtValues.push_back(et);
                caloTopoTowerEtaValues.push_back(cluster->eta());
                caloTopoTowerPhiValues.push_back(cluster->phi());

                if (et < 0) continue; // don't store to digitized memories
            }
            caloTapoTowerOff.endEvent(caloTopoTowerEtValues.size());

            unsigned int topocluster422_it = 0;
            // Loop over the clusters and store Et, Eta, Phi
            for (const auto* cluster : *CaloTopoClusters422) {
                if (!cluster) continue;

                double et = cluster->e() / cosh(cluster->eta()) / 1000.0; // Et in GeV
                topo422EtValues.push_back(et);
                topo422EtaValues.push_back(cluster->eta());
                topo422PhiValues.push_back(cluster->phi());

                if (et < 0) continue; // FIXME for now don't store negative Et to digitized memories
            }
            //end of this event's towers
            topo422Off.endEvent(topo422EtaValues.size());

            // Temporary vector for sorting by Et
            std::vector<std::pair<size_t, double>> jFexSRJetEtWithIndex;

            for (size_t i = 0; i < L1_jFexSRJetRoI->size(); ++i) {
                const auto& jet = (*L1_jFexSRJetRoI)[i];
                double et = jet->et() / 1000.0; // Already in GeV
                jFexSRJetEtWithIndex.emplace_back(i, et);
                //std::cout << "i : " << i << " and Et : " << et << "\n";
            }

            // Sort descending by Et
            std::sort(jFexSRJetEtWithIndex.begin(), jFexSRJetEtWithIndex.end(),
                    [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                        return a.second > b.second;
                    });

            // Fill vectors in sorted order
            unsigned int jfex_it = 0;
            for (const auto& [index, et] : jFexSRJetEtWithIndex) {
                //std::cout << "index: " << index << "\n";
                //std::cout << "et : " << et << "\n";
                const auto& jet = (*L1_jFexSRJetRoI)[index];

                jFexSRJEtIndexValues.push_back(static_cast<unsigned int>(index));
                jFexSRJEtValues.push_back(et);
                jFexSRJEtaValues.push_back(jet->eta());
                jFexSRJPhiValues.push_back(jet->phi());

                if (et < 0) continue; // FIXME
            }

            // Leading jet
            if (!jFexSRJetEtWithIndex.empty()) {
                const auto& leading = (*L1_jFexSRJetRoI)[jFexSRJetEtWithIndex[0].first];
                jFexSRJLeadingEtValues.push_back(jFexSRJEtValues[0]);
                jFexSRJLeadingEtaValues.push_back(leading->eta());
                jFexSRJLeadingPhiValues.push_back(leading->phi());
            }

            // Subleading jet
            if (jFexSRJetEtWithIndex.size() > 1) {
                const auto& subleading = (*L1_jFexSRJetRoI)[jFexSRJetEtWithIndex[1].first];
                jFexSRJSubleadingEtValues.push_back(jFexSRJEtValues[1]);
                jFexSRJSubleadingEtaValues.push_back(subleading->eta());
                jFexSRJSubleadingPhiValues.push_back(subleading->phi());
            }
            
            //end of this event
            jFexSRJOff.endEvent(jFexSRJEtValues.size());

            // Temporary vector for sorting by Et
            std::vector<std::pair<size_t, double>> jFexLRJetEtWithIndex;

            for (size_t i = 0; i < L1_jFexLRJetRoI->size(); ++i) {
                const auto& jet = (*L1_jFexLRJetRoI)[i];
                double et = jet->et() / 1000.0; // Already in GeV
                jFexLRJetEtWithIndex.emplace_back(i, et);
            }

            // Sort descending by Et
            std::sort(jFexLRJetEtWithIndex.begin(), jFexLRJetEtWithIndex.end(),
                    [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                        return a.second > b.second;
                    });

            // Fill vectors in sorted order
            for (const auto& [index, et] : jFexLRJetEtWithIndex) {
                const auto& jet = (*L1_jFexLRJetRoI)[index];

                jFexLRJEtIndexValues.push_back(static_cast<unsigned int>(index));
                jFexLRJEtValues.push_back(et);
                jFexLRJEtaValues.push_back(jet->eta());
                jFexLRJPhiValues.push_back(jet->phi());

            }

            // Leading jet
            if (!jFexLRJetEtWithIndex.empty()) {
                const auto& leading = (*L1_jFexLRJetRoI)[jFexLRJetEtWithIndex[0].first];
                jFexLRJLeadingEtValues.push_back(jFexLRJEtValues[0]);
                jFexLRJLeadingEtaValues.push_back(leading->eta());
                jFexLRJLeadingPhiValues.push_back(leading->phi());
            }

            // Subleading jet
            if (jFexLRJetEtWithIndex.size() > 1) {
                const auto& subleading = (*L1_jFexLRJetRoI)[jFexLRJetEtWithIndex[1].first];
                jFexLRJSubleadingEtValues.push_back(jFexLRJEtValues[1]);
                jFexLRJSubleadingEtaValues.push_back(subleading->eta());
                jFexLRJSubleadingPhiValues.push_back(subleading->phi());
            }

            // mark end of event
            jFexLRJOff.endEvent(jFexLRJEtaValues.size());

            // Temporary vector for sorting by Et
            std::vector<std::pair<size_t, double>> gFexSRJetEtWithIndex;

            for (size_t i = 0; i < L1_gFexSRJetRoI->size(); ++i) {
                const auto& jet = (*L1_gFexSRJetRoI)[i];
                double et = jet->et() / 1000.0; // Already in GeV
                gFexSRJetEtWithIndex.emplace_back(i, et);
            }

            // Sort descending by Et
            std::sort(gFexSRJetEtWithIndex.begin(), gFexSRJetEtWithIndex.end(),
                    [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                        return a.second > b.second;
                    });

            // Fill vectors in sorted order
            unsigned int gfex_it = 0;
            for (const auto& [index, et] : gFexSRJetEtWithIndex) {
                const auto& jet = (*L1_gFexSRJetRoI)[index];

                gFexSRJEtIndexValues.push_back(static_cast<unsigned int>(index));
                gFexSRJEtValues.push_back(et);
                gFexSRJEtaValues.push_back(jet->eta());
                gFexSRJPhiValues.push_back(jet->phi());

                if (et < 0) continue;
            }

            // Leading jet
            if (!gFexSRJetEtWithIndex.empty()) {
                const auto& leading = (*L1_gFexSRJetRoI)[gFexSRJetEtWithIndex[0].first];
                gFexSRJLeadingEtValues.push_back(gFexSRJEtValues[0]);
                gFexSRJLeadingEtaValues.push_back(leading->eta());
                gFexSRJLeadingPhiValues.push_back(leading->phi());
            }

            // Subleading jet
            if (gFexSRJetEtWithIndex.size() > 1) {
                const auto& subleading = (*L1_gFexSRJetRoI)[gFexSRJetEtWithIndex[1].first];
                gFexSRJSubleadingEtValues.push_back(gFexSRJEtValues[1]);
                gFexSRJSubleadingEtaValues.push_back(subleading->eta());
                gFexSRJSubleadingPhiValues.push_back(subleading->phi());
            }

            // mark end of event
            gFexSRJOff.endEvent(gFexSRJEtValues.size());

            // Temporary vector for sorting by Et
            std::vector<std::pair<size_t, double>> gFexLRJetEtWithIndex;

            for (size_t i = 0; i < L1_gFexLRJetRoI->size(); ++i) {
                const auto& jet = (*L1_gFexLRJetRoI)[i];
                double et = jet->et() / 1000.0; // already in GeV
                gFexLRJetEtWithIndex.emplace_back(i, et);
            }

            // Sort descending by Et
            std::sort(gFexLRJetEtWithIndex.begin(), gFexLRJetEtWithIndex.end(),
                    [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                        return a.second > b.second;
                    });

            // Fill vectors in sorted order
            for (const auto& [index, et] : gFexLRJetEtWithIndex) {
                const auto& jet = (*L1_gFexLRJetRoI)[index];

                gFexLRJEtIndexValues.push_back(static_cast<unsigned int>(index));
                gFexLRJEtValues.push_back(et);
                gFexLRJEtaValues.push_back(jet->eta());
                gFexLRJPhiValues.push_back(jet->phi());
            }

            // Leading jet
            if (!gFexLRJetEtWithIndex.empty()) {
                const auto& leading = (*L1_gFexLRJetRoI)[gFexLRJetEtWithIndex[0].first];
                gFexLRJLeadingEtValues.push_back(gFexLRJEtValues[0]);
                gFexLRJLeadingEtaValues.push_back(leading->eta());
                gFexLRJLeadingPhiValues.push_back(leading->phi());
            }

            // Subleading jet
            if (gFexLRJetEtWithIndex.size() > 1) {
                const auto& subleading = (*L1_gFexLRJetRoI)[gFexLRJetEtWithIndex[1].first];
                gFexLRJSubleadingEtValues.push_back(gFexLRJEtValues[1]);
                gFexLRJSubleadingEtaValues.push_back(subleading->eta());
                gFexLRJSubleadingPhiValues.push_back(subleading->phi());
            }

            // mark end of event
            gFexLRJOff.endEvent(gFexLRJEtValues.size());

            std::vector<std::pair<size_t, double>> hltJetEtWithIndex;
            for (size_t i = 0; i < HLT_AntiKt4EMTopoJets_subjesIS->size(); ++i) {
                const auto& el = (*HLT_AntiKt4EMTopoJets_subjesIS)[i];
                double et = el->e() / (1000.0 * cosh(el->eta()));
                hltJetEtWithIndex.emplace_back(i, et);  // Store index and Et for sorting
            }
            // Sort by descending Et
            std::sort(hltJetEtWithIndex.begin(), hltJetEtWithIndex.end(),
                    [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                        return a.second > b.second;
                    });
            // Now push back into vectors in sorted order
            for (const auto& [index, et] : hltJetEtWithIndex) {
                const auto& el = (*HLT_AntiKt4EMTopoJets_subjesIS)[index];
                hltAntiKt4SRJEtIndexValues.push_back(static_cast<unsigned int>(index));
                hltAntiKt4SRJEtValues.push_back(et);
                hltAntiKt4SRJEtaValues.push_back(el->eta());
                hltAntiKt4SRJPhiValues.push_back(el->phi());
            }

            // Store leading and subleading jets if available
            if (!hltJetEtWithIndex.empty()) {
                const auto& leading = (*HLT_AntiKt4EMTopoJets_subjesIS)[hltJetEtWithIndex[0].first];
                hltAntiKt4SRJLeadingEtValues.push_back(hltAntiKt4SRJEtValues[0]);
                hltAntiKt4SRJLeadingEtaValues.push_back(leading->eta());
                hltAntiKt4SRJLeadingPhiValues.push_back(leading->phi());
            }

            if (hltJetEtWithIndex.size() > 1) {
                const auto& subleading = (*HLT_AntiKt4EMTopoJets_subjesIS)[hltJetEtWithIndex[1].first];
                hltAntiKt4SRJSubleadingEtValues.push_back(hltAntiKt4SRJEtValues[1]);
                hltAntiKt4SRJSubleadingEtaValues.push_back(subleading->eta());
                hltAntiKt4SRJSubleadingPhiValues.push_back(subleading->phi());
            }

            // mark end of event
            hltAntiKt4SRJOff.endEvent(hltAntiKt4SRJEtValues.size());

            // --- Loop over L1_gFexLRJetRoI ---
            //for (const auto& el : *gFexLRJets) {
            //    gfex_larger_jet_pt_values.push_back(el->et() / 1000.0);
            //    float gfex_larger_jet_Et = el->et() / 1000.0;
            //}

            // --- Loop over InTimeAntiKt4TruthJets ---
            // Temporary vector to hold (index, Et) for sorting
            std::vector<std::pair<size_t, double>> truthJetEtWithIndex;

            for (size_t i = 0; i < InTimeAntiKt4TruthJets->size(); ++i) {
                const auto& jet = (*InTimeAntiKt4TruthJets)[i];
                double et = jet->e() / (1000.0 * cosh(jet->eta()));
                truthJetEtWithIndex.emplace_back(i, et);  // Store original index and Et
            }

            // Sort by descending Et
            std::sort(truthJetEtWithIndex.begin(), truthJetEtWithIndex.end(),
                    [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                        return a.second > b.second;
                    });

            // Fill vectors in sorted order
            for (const auto& [index, et] : truthJetEtWithIndex) {
                const auto& jet = (*InTimeAntiKt4TruthJets)[index];
                inTimeAntiKt4TruthSRJEtIndexValues.push_back(static_cast<unsigned int>(index));
                inTimeAntiKt4TruthSRJEtValues.push_back(et);
                inTimeAntiKt4TruthSRJEtaValues.push_back(jet->eta());
                inTimeAntiKt4TruthSRJPhiValues.push_back(jet->phi());
            }

            // Store leading jet info if available
            if (!truthJetEtWithIndex.empty()) {
                const auto& leading = (*InTimeAntiKt4TruthJets)[truthJetEtWithIndex[0].first];
                inTimeAntiKt4TruthSRJLeadingEtValues.push_back(inTimeAntiKt4TruthSRJEtValues[0]);
                inTimeAntiKt4TruthSRJLeadingEtaValues.push_back(leading->eta());
                inTimeAntiKt4TruthSRJLeadingPhiValues.push_back(leading->phi());
            }

            // Store subleading jet info if available
            if (truthJetEtWithIndex.size() > 1) {
                const auto& subleading = (*InTimeAntiKt4TruthJets)[truthJetEtWithIndex[1].first];
                inTimeAntiKt4TruthSRJSubleadingEtValues.push_back(inTimeAntiKt4TruthSRJEtValues[1]);
                inTimeAntiKt4TruthSRJSubleadingEtaValues.push_back(subleading->eta());
                inTimeAntiKt4TruthSRJSubleadingPhiValues.push_back(subleading->phi());
            }

            // mark end of event
            inTimeAntiKt4TruthSRJOff.endEvent(inTimeAntiKt4TruthSRJEtValues.size());

            // Temporary vector to hold (index, Et) for sorting
            std::vector<std::pair<size_t, double>> recoJetEtWithIndex;

            for (size_t i = 0; i < AntiKt10UFOCSSKJets->size(); ++i) {
                const auto& jet = (*AntiKt10UFOCSSKJets)[i];
                double et = jet->e() / (1000.0 * cosh(jet->eta()));
                recoJetEtWithIndex.emplace_back(i, et);
            }

            // Sort by descending Et
            std::sort(recoJetEtWithIndex.begin(), recoJetEtWithIndex.end(),
                    [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                        return a.second > b.second;
                    });

            // Fill vectors in sorted order
            for (const auto& [index, et] : recoJetEtWithIndex) {
                const auto& jet = (*AntiKt10UFOCSSKJets)[index];
                recoAntiKt10LRJEtIndexValues.push_back(static_cast<unsigned int>(index));
                recoAntiKt10LRJEtValues.push_back(et);
                recoAntiKt10LRJEtaValues.push_back(jet->eta());
                recoAntiKt10LRJPhiValues.push_back(jet->phi());
                recoAntiKt10LRJMassValues.push_back(jet->m() / 1000.0);
            }

            // Leading jet
            if (!recoJetEtWithIndex.empty()) {
                const auto& leading = (*AntiKt10UFOCSSKJets)[recoJetEtWithIndex[0].first];
                recoAntiKt10LRJLeadingEtValues.push_back(recoAntiKt10LRJEtValues[0]);
                recoAntiKt10LRJLeadingEtaValues.push_back(leading->eta());
                recoAntiKt10LRJLeadingPhiValues.push_back(leading->phi());
                recoAntiKt10LRJLeadingMassValues.push_back(leading->m() / 1000.0);
            }

            // Subleading jet
            if (recoJetEtWithIndex.size() > 1) {
                const auto& subleading = (*AntiKt10UFOCSSKJets)[recoJetEtWithIndex[1].first];
                recoAntiKt10LRJSubleadingEtValues.push_back(recoAntiKt10LRJEtValues[1]);
                recoAntiKt10LRJSubleadingEtaValues.push_back(subleading->eta());
                recoAntiKt10LRJSubleadingPhiValues.push_back(subleading->phi());
                recoAntiKt10LRJSubleadingMassValues.push_back(subleading->m() / 1000.0);
            }

            recoAntiKt10LRJOff.endEvent(recoAntiKt10LRJEtValues.size());

            // Temporary vector to hold (index, Et) for sorting
            std::vector<std::pair<size_t, double>> truthWZJetEtWithIndex;

            for (size_t i = 0; i < AntiKt4TruthDressedWZJets->size(); ++i) {
                const auto& jet = (*AntiKt4TruthDressedWZJets)[i];
                double et = jet->e() / (1000.0 * cosh(jet->eta()));
                truthWZJetEtWithIndex.emplace_back(i, et);
            }

            // Sort by descending Et
            std::sort(truthWZJetEtWithIndex.begin(), truthWZJetEtWithIndex.end(),
                    [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
                        return a.second > b.second;
                    });

            // Fill vectors in sorted order
            for (const auto& [index, et] : truthWZJetEtWithIndex) {
                const auto& jet = (*AntiKt4TruthDressedWZJets)[index];
                truthAntiKt4WZSRJEtIndexValues.push_back(static_cast<unsigned int>(index));
                truthAntiKt4WZSRJEtValues.push_back(et);
                truthAntiKt4WZSRJEtaValues.push_back(jet->eta());
                truthAntiKt4WZSRJPhiValues.push_back(jet->phi());
                truthAntiKt4WZSRJMassValues.push_back(jet->m() / 1000.0);
            }

            // Leading jet
            if (!truthWZJetEtWithIndex.empty()) {
                const auto& leading = (*AntiKt4TruthDressedWZJets)[truthWZJetEtWithIndex[0].first];
                truthAntiKt4WZSRJLeadingEtValues.push_back(truthAntiKt4WZSRJEtValues[0]);
                truthAntiKt4WZSRJLeadingEtaValues.push_back(leading->eta());
                truthAntiKt4WZSRJLeadingPhiValues.push_back(leading->phi());
                truthAntiKt4WZSRJLeadingMassValues.push_back(leading->m() / 1000.0);

            }

            // Subleading jet
            if (truthWZJetEtWithIndex.size() > 1) {
                const auto& subleading = (*AntiKt4TruthDressedWZJets)[truthWZJetEtWithIndex[1].first];
                truthAntiKt4WZSRJSubleadingEtValues.push_back(truthAntiKt4WZSRJEtValues[1]);
                truthAntiKt4WZSRJSubleadingEtaValues.push_back(subleading->eta());
                truthAntiKt4WZSRJSubleadingPhiValues.push_back(subleading->phi());
                truthAntiKt4WZSRJSubleadingMassValues.push_back(subleading->m() / 1000.0);
            }
            truthAntiKt4Off.endEvent(truthAntiKt4WZSRJEtValues.size());
        } // loop through events
        std::cout << "for jz: " << jzSlice << " these many events passed: " << passedEventsCounter << " out of: " << event.getEntries() << "\n";
        std::cout << " these many events skipped due to empty truth container: " << skippedEventsEmptyTruth << " these many events skipped due to pt hard < pt pileup: " << skippedEventsHSTP << "\n";
        std::cout << "closing files" << "\n";
        f->Close();
    } // loop through filenames

    // --- End of event processing timing ---
    auto tEndProcessing = std::chrono::steady_clock::now();
    std::chrono::duration<double> procTime = tEndProcessing - tStartProcessing;
    double timePerEvent = (totalEventsLooped > 0)
        ? procTime.count() / static_cast<double>(totalEventsLooped)
        : 0.0;

    std::cout << "\n[Timing] Event processing total: " << procTime.count() << " s\n"
              << "[Timing] Events processed (looped over): " << totalEventsLooped << "\n"
              << "[Timing] Avg time per event: " << timePerEvent << " s/event\n";

    // --- Now time the ROOT writing separately ---
    auto tStartWrite = std::chrono::steady_clock::now();

    // get the H5 file
    H5::H5File h5("/data/crystalwang/testVectorPipeline/testData/outputHDF5Files/exampleHDF5.h5", H5F_ACC_TRUNC);

    // Create all parent groups needed by your dataset paths
    //h5.createGroup("/EventInfo");

    //h5.createGroup("/truthHiggs");
    //h5.createGroup("/truthHiggs/Et");

    //h5.createGroup("/truthB");
    //h5.createGroup("/truthB/Et");

    make_group(h5, "CaloTopoTowers");
    make_group(h5, "topo422");
    make_group(h5, "gFEXSRJ");
    make_group(h5, "gFEXLRJ");
    make_group(h5, "jFEXSRJ");
    make_group(h5, "jFEXLRJ");
    make_group(h5, "HLTJets");
    make_group(h5, "RecoAK10");
    make_group(h5, "InTimeAntiKt4TruthJets");
    make_group(h5, "AntiKt4TruthDressedWZJets");
    h5.createGroup("/EventInfo");

        // ---- Event info group ----
    write_vector(h5, "/EventInfo/mcEventWeight", mcEventWeights);
    //sumofweights is a scalar so wrap in vector
    std::vector<double> sumOfWeightsVec(1, sumOfWeightsForSample);
    write_vector(h5, "/EventInfo/sumOfWeightsForSample", sumOfWeightsVec);
    write_vector(h5, "/EventInfo/eventWeights",   eventWeights);
    std::vector<double> sampleJZVec(1, sampleJZSlice);
    write_vector(h5, "/EventInfo/sampleJZSlice", sampleJZVec);

    // ---- Truth Higgs ----
    /**write_vector(h5, "/truthHiggs/Et/data",      truthHiggsEtValues);
    write_vector(h5, "/truthHiggs/Eta/data",     truthHiggsEtaValues);
    write_vector(h5, "/truthHiggs/Phi/data",     truthHiggsPhiValues);
    write_vector(h5, "/truthHiggs/pT/data",      truthHiggspTValues);
    write_vector(h5, "/truthHiggs/px/data",      truthHiggspxValues);
    write_vector(h5, "/truthHiggs/py/data",      truthHiggspyValues);
    write_vector(h5, "/truthHiggs/pz/data",      truthHiggspzValues);
    write_vector(h5, "/truthHiggs/Energy/data",  truthHiggsEnergyValues);
    write_vector(h5, "/truthHiggs/invMass/data", truthHiggsInvMassValues);
    write_vector(h5, "/truthHiggs/index/data",   indexOfHiggsValues);
    write_vector(h5, "/truthHiggs/Et/offsets",   truthHiggsOff.offsets);

    // ---- Truth b-quarks ----
    write_vector(h5, "/truthB/higgsIndex/data",  higgsIndexValues);
    write_vector(h5, "/truthB/Et/data",          truthbquarksEtValues);
    write_vector(h5, "/truthB/Eta/data",         truthbquarksEtaValues);
    write_vector(h5, "/truthB/Phi/data",         truthbquarksPhiValues);
    write_vector(h5, "/truthB/pT/data",          truthbquarkspTValues);
    write_vector(h5, "/truthB/px/data",          truthbquarkspxValues);
    write_vector(h5, "/truthB/py/data",          truthbquarkspyValues);
    write_vector(h5, "/truthB/pz/data",          truthbquarkspzValues);
    write_vector(h5, "/truthB/Energy/data",      truthbquarksEnergyValues);
    write_vector(h5, "/truthB/Et/offsets",       truthBOff.offsets);
    */

    // ---- Calo towers ----
    write_vector(h5, "/CaloTopoTowers/Et/data",     caloTopoTowerEtValues);
    write_vector(h5, "/CaloTopoTowers/Eta/data",    caloTopoTowerEtaValues);
    write_vector(h5, "/CaloTopoTowers/Phi/data",    caloTopoTowerPhiValues);
    write_vector(h5, "/CaloTopoTowers/Et/offsets",  caloTapoTowerOff.offsets);

    // ---- TopoClusters422 ----
    write_vector(h5, "/topo422/Et/data",    topo422EtValues);
    write_vector(h5, "/topo422/Eta/data",   topo422EtaValues);
    write_vector(h5, "/topo422/Phi/data",   topo422PhiValues);
    write_vector(h5, "/topo422/Et/offsets", topo422Off.offsets);

    // ---- InTimeAntiKt4TruthJets ----
    write_vector(h5, "/InTimeAntiKt4TruthJets/EtIndex/data", inTimeAntiKt4TruthSRJEtIndexValues);
    write_vector(h5, "/InTimeAntiKt4TruthJets/Et/data",      inTimeAntiKt4TruthSRJEtValues);
    write_vector(h5, "/InTimeAntiKt4TruthJets/Eta/data",     inTimeAntiKt4TruthSRJEtaValues);
    write_vector(h5, "/InTimeAntiKt4TruthJets/Phi/data",     inTimeAntiKt4TruthSRJPhiValues);
    write_vector(h5, "/InTimeAntiKt4TruthJets/Et/offsets",   inTimeAntiKt4TruthSRJOff.offsets);

    write_vector(h5, "/InTimeAntiKt4TruthJets/LeadingEt",   inTimeAntiKt4TruthSRJLeadingEtValues);
    write_vector(h5, "/InTimeAntiKt4TruthJets/LeadingEta",  inTimeAntiKt4TruthSRJLeadingEtaValues);
    write_vector(h5, "/InTimeAntiKt4TruthJets/LeadingPhi",  inTimeAntiKt4TruthSRJLeadingPhiValues);
    write_vector(h5, "/InTimeAntiKt4TruthJets/SubleadingEt",   inTimeAntiKt4TruthSRJSubleadingEtValues);
    write_vector(h5, "/InTimeAntiKt4TruthJets/SubleadingEta",  inTimeAntiKt4TruthSRJSubleadingEtaValues);
    write_vector(h5, "/InTimeAntiKt4TruthJets/SubleadingPhi",  inTimeAntiKt4TruthSRJSubleadingPhiValues);

    // ---- truth dressed WZ jets ----
    write_vector(h5, "/AntiKt4TruthDressedWZJets/EtIndex/data", truthAntiKt4WZSRJEtIndexValues);
    write_vector(h5, "/AntiKt4TruthDressedWZJets/Et/data",      truthAntiKt4WZSRJEtValues);
    write_vector(h5, "/AntiKt4TruthDressedWZJets/Eta/data",     truthAntiKt4WZSRJEtaValues);
    write_vector(h5, "/AntiKt4TruthDressedWZJets/Phi/data",     truthAntiKt4WZSRJPhiValues);
    h5.createGroup("/AntiKt4TruthDressedWZJets/Mass");
    write_vector(h5, "/AntiKt4TruthDressedWZJets/Mass/data",    truthAntiKt4WZSRJMassValues);
    write_vector(h5, "/AntiKt4TruthDressedWZJets/Et/offsets",   truthAntiKt4Off.offsets);

    write_vector(h5, "/AntiKt4TruthDressedWZJets/LeadingEt",    truthAntiKt4WZSRJLeadingEtValues);
    write_vector(h5, "/AntiKt4TruthDressedWZJets/LeadingEta",   truthAntiKt4WZSRJLeadingEtaValues);
    write_vector(h5, "/AntiKt4TruthDressedWZJets/LeadingPhi",   truthAntiKt4WZSRJLeadingPhiValues);
    write_vector(h5, "/AntiKt4TruthDressedWZJets/LeadingMass",  truthAntiKt4WZSRJLeadingMassValues);
    write_vector(h5, "/AntiKt4TruthDressedWZJets/SubleadingEt",   truthAntiKt4WZSRJSubleadingEtValues);
    write_vector(h5, "/AntiKt4TruthDressedWZJets/SubleadingEta",  truthAntiKt4WZSRJSubleadingEtaValues);
    write_vector(h5, "/AntiKt4TruthDressedWZJets/SubleadingPhi",  truthAntiKt4WZSRJSubleadingPhiValues);
    write_vector(h5, "/AntiKt4TruthDressedWZJets/SubleadingMass", truthAntiKt4WZSRJSubleadingMassValues);


    // FIX ME ALL VAR NAMES BELOW 
    // ---- gFex SR ----
    write_vector(h5, "/gFEXSRJ/EtIndex/data", gFexSRJEtIndexValues);
    write_vector(h5, "/gFEXSRJ/Et/data", gFexSRJEtValues);
    write_vector(h5, "/gFEXSRJ/Eta/data", gFexSRJEtaValues);
    write_vector(h5, "/gFEXSRJ/Phi/data", gFexSRJPhiValues);
    write_vector(h5, "/gFEXSRJ/Et/offsets", gFexSRJOff.offsets);

    write_vector(h5, "/gFEXSRJ/LeadingEt", gFexSRJLeadingEtValues);
    write_vector(h5, "/gFEXSRJ/LeadingEta", gFexSRJLeadingEtaValues);
    write_vector(h5, "/gFEXSRJ/LeadingPhi", gFexSRJLeadingPhiValues);
    write_vector(h5, "/gFEXSRJ/SubleadingEt", gFexSRJSubleadingEtValues);
    write_vector(h5, "/gFEXSRJ/SubleadingEta", gFexSRJSubleadingEtaValues);
    write_vector(h5, "/gFEXSRJ/SubleadingPhi", gFexSRJSubleadingPhiValues);

    // ---- gFex LR ----
    write_vector(h5, "/gFEXLRJ/EtIndex/data", gFexLRJEtIndexValues);
    write_vector(h5, "/gFEXLRJ/Et/data",      gFexLRJEtValues);
    write_vector(h5, "/gFEXLRJ/Eta/data",     gFexLRJEtaValues);
    write_vector(h5, "/gFEXLRJ/Phi/data",     gFexLRJPhiValues);
    write_vector(h5, "/gFEXLRJ/Et/offsets",   gFexLRJOff.offsets);

    write_vector(h5, "/gFEXLRJ/LeadingEt",    gFexLRJLeadingEtValues);
    write_vector(h5, "/gFEXLRJ/LeadingEta",   gFexLRJLeadingEtaValues);
    write_vector(h5, "/gFEXLRJ/LeadingPhi",   gFexLRJLeadingPhiValues);
    write_vector(h5, "/gFEXLRJ/SubleadingEt", gFexLRJSubleadingEtValues);
    write_vector(h5, "/gFEXLRJ/SubleadingEta",gFexLRJSubleadingEtaValues);
    write_vector(h5, "/gFEXLRJ/SubleadingPhi",gFexLRJSubleadingPhiValues);

    // ---- jFex SR ----
    write_vector(h5, "/jFEXSRJ/EtIndex/data", jFexSRJEtIndexValues);
    write_vector(h5, "/jFEXSRJ/Et/data",      jFexSRJEtValues);
    write_vector(h5, "/jFEXSRJ/Eta/data",     jFexSRJEtaValues);
    write_vector(h5, "/jFEXSRJ/Phi/data",     jFexSRJPhiValues);
    write_vector(h5, "/jFEXSRJ/Et/offsets",   jFexSRJOff.offsets);

    write_vector(h5, "/jFEXSRJ/LeadingEt",    jFexSRJLeadingEtValues);
    write_vector(h5, "/jFEXSRJ/LeadingEta",   jFexSRJLeadingEtaValues);
    write_vector(h5, "/jFEXSRJ/LeadingPhi",   jFexSRJLeadingPhiValues);
    write_vector(h5, "/jFEXSRJ/SubleadingEt", jFexSRJSubleadingEtValues);
    write_vector(h5, "/jFEXSRJ/SubleadingEta",jFexSRJSubleadingEtaValues);
    write_vector(h5, "/jFEXSRJ/SubleadingPhi",jFexSRJSubleadingPhiValues);

    // ---- jFex LR ----
    write_vector(h5, "/jFEXLRJ/EtIndex/data", jFexLRJEtIndexValues);
    write_vector(h5, "/jFEXLRJ/Et/data",      jFexLRJEtValues);
    write_vector(h5, "/jFEXLRJ/Eta/data",     jFexLRJEtaValues);
    write_vector(h5, "/jFEXLRJ/Phi/data",     jFexLRJPhiValues);
    write_vector(h5, "/jFEXLRJ/Et/offsets",   jFexLRJOff.offsets);

    write_vector(h5, "/jFEXLRJ/LeadingEt",    jFexLRJLeadingEtValues);
    write_vector(h5, "/jFEXLRJ/LeadingEta",   jFexLRJLeadingEtaValues);
    write_vector(h5, "/jFEXLRJ/LeadingPhi",   jFexLRJLeadingPhiValues);
    write_vector(h5, "/jFEXLRJ/SubleadingEt", jFexLRJSubleadingEtValues);
    write_vector(h5, "/jFEXLRJ/SubleadingEta",jFexLRJSubleadingEtaValues);
    write_vector(h5, "/jFEXLRJ/SubleadingPhi",jFexLRJSubleadingPhiValues);

    // ---- HLT AntiKt4EMTopo jets ----
    write_vector(h5, "/HLTJets/EtIndex/data", hltAntiKt4SRJEtIndexValues);
    write_vector(h5, "/HLTJets/Et/data",      hltAntiKt4SRJEtValues);
    write_vector(h5, "/HLTJets/Eta/data",     hltAntiKt4SRJEtaValues);
    write_vector(h5, "/HLTJets/Phi/data",     hltAntiKt4SRJPhiValues);
    write_vector(h5, "/HLTJets/Et/offsets",   hltAntiKt4SRJOff.offsets);

    write_vector(h5, "/HLTJets/LeadingEt",    hltAntiKt4SRJLeadingEtValues);
    write_vector(h5, "/HLTJets/LeadingEta",   hltAntiKt4SRJLeadingEtaValues);
    write_vector(h5, "/HLTJets/LeadingPhi",   hltAntiKt4SRJLeadingPhiValues);
    write_vector(h5, "/HLTJets/SubleadingEt", hltAntiKt4SRJSubleadingEtValues);
    write_vector(h5, "/HLTJets/SubleadingEta",hltAntiKt4SRJSubleadingEtaValues);
    write_vector(h5, "/HLTJets/SubleadingPhi",hltAntiKt4SRJSubleadingPhiValues);

    // ---- Reco AntiKt10UFOCSSK jets ----
    write_vector(h5, "/RecoAK10/EtIndex/data", recoAntiKt10LRJEtIndexValues);
    write_vector(h5, "/RecoAK10/Et/data",      recoAntiKt10LRJEtValues);
    write_vector(h5, "/RecoAK10/Eta/data",     recoAntiKt10LRJEtaValues);
    write_vector(h5, "/RecoAK10/Phi/data",     recoAntiKt10LRJPhiValues);
    h5.createGroup("/RecoAK10/Mass");
    write_vector(h5, "/RecoAK10/Mass/data",    recoAntiKt10LRJMassValues);
    write_vector(h5, "/RecoAK10/Et/offsets",   recoAntiKt10LRJOff.offsets);

    write_vector(h5, "/RecoAK10/LeadingEt",    recoAntiKt10LRJLeadingEtValues);
    write_vector(h5, "/RecoAK10/LeadingEta",   recoAntiKt10LRJLeadingEtaValues);
    write_vector(h5, "/RecoAK10/LeadingPhi",   recoAntiKt10LRJLeadingPhiValues);
    write_vector(h5, "/RecoAK10/LeadingMass",  recoAntiKt10LRJLeadingMassValues);
    write_vector(h5, "/RecoAK10/SubleadingEt", recoAntiKt10LRJSubleadingEtValues);
    write_vector(h5, "/RecoAK10/SubleadingEta",recoAntiKt10LRJSubleadingEtaValues);
    write_vector(h5, "/RecoAK10/SubleadingPhi",recoAntiKt10LRJSubleadingPhiValues);
    write_vector(h5, "/RecoAK10/SubleadingMass",recoAntiKt10LRJSubleadingMassValues);
    
    // std::cout << "Wrote HDF5 file: " << outFile << "\n";
    auto tEndWrite = std::chrono::steady_clock::now();
    std::chrono::duration<double> writeTime = tEndWrite - tStartWrite;

    std::cout << "[Timing] HDF5 file writing time: "
              << writeTime.count() << " s\n";
}

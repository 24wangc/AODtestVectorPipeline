// To execute: e.g., root ; .L nTupler.C ; nTupler(true, true, true) 
#include <algorithm>
#include <numeric>   // std::iota

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
#include "analysisHelperFunctions.h"

#include <sstream>
#include <bitset>
#include <iomanip>
#include <cstdint>
#include <chrono>

// Used for digitized data writing
struct OutputFiles {
    std::string topo422;
    std::string caloTopoTowers;
    std::string gFex;
    std::string jFex;
};

// Settings
const bool afBool = true;
const bool vbfBool = true;
const unsigned int nJZSlices = 10;

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


// Helper to construct memprint / test vector filenames
inline OutputFiles makeMemPrintFilenames(bool signalBool, bool vbfBool, int jzSlice) {
    static const std::string base = "/data/crystalwang/testVectorPipeline/testData/testVectors";

    // ensure subdirs exist
    namespace fs = std::filesystem;
    fs::create_directories(base + "/CaloTopo_422");
    fs::create_directories(base + "/CaloTopoTowers");
    fs::create_directories(base + "/gFex");
    fs::create_directories(base + "/jFex");

    std::string tag;
    if (signalBool) {
        tag = vbfBool ? "SIGNAL_HHBBBB_VBF" : "HHbbbb_HLLHC";
    } else {
        if (jzSlice < 0 || jzSlice > 9)
            throw std::out_of_range("jzSlice must be in [0,9]");
        tag = "jj_JZ" + std::to_string(jzSlice);
    }

    OutputFiles out;
    out.topo422        = base + "/CaloTopo_422/"    + tag + "_topo422.dat";
    out.caloTopoTowers = base + "/CaloTopoTowers/"  + tag + "_calotopotowers.dat";
    out.gFex           = base + "/gFex/"            + tag + "_gfex_smallrj.dat";
    out.jFex           = base + "/jFex/"            + tag + "_jfex_smallrj.dat";
    return out;
}

// structure for a  collection containing the vectors for each kinematic var
struct collTree {
    std::vector<double>* Et {nullptr};
    std::vector<double>* Eta {nullptr};
    std::vector<double>* Phi {nullptr};
};

// connect the branch addresses for Et Eta and Phi for the collTree
inline void branchify(TTree* tree, collTree& v) {
    tree->SetBranchAddress("Et", &v.Et);
    tree->SetBranchAddress("Eta", &v.Eta);
    tree->SetBranchAddress("Phi", &v.Phi);
}

//digitize
unsigned int digitize(double value, int bit_length, double min_val, double max_val) {
    // Check if value is in range
    if (value < min_val) {
        value = min_val;
        std::cout << "Warning: Value " << value
          << " is out of range (" << min_val
          << ", " << max_val << ")\n";
    }
    if (value > max_val){
        value = max_val;
        std::cout << "Warning: Value " << value
          << " is out of range (" << min_val
          << ", " << max_val << ")\n";
    }

    double scale = (std::pow(2, bit_length) - 1) / (max_val - min_val);
    return static_cast<unsigned int>(std::round((value - min_val) * scale));
}

// ditize and write test vectors
inline void writeTestVectors(std::ofstream& out, Long64_t iEvt, const collTree& v) {
    unsigned int idx = 0;

    // loop through the events of this tree
    for (size_t i = 0; i < v.Et->size(); ++i) {
        double et  = (*v.Et)[i];
        double eta = (*v.Eta)[i];
        double phi = (*v.Phi)[i];

        if (et < 0) continue;  // keep your cut if you want

        // digitize all the variables
        unsigned int et_bin  = digitize(et,  et_bit_length_,  et_min_,  et_max_);
        unsigned int eta_bin = digitize(eta, eta_bit_length_, eta_min_, eta_max_);
        unsigned int phi_bin = digitize(phi, phi_bit_length_, phi_min_, phi_max_);

        // print these variablese to the output file
        std::stringstream binary_ss;
        binary_ss << std::bitset<et_bit_length_>(et_bin)   << "|"
                  << std::bitset<eta_bit_length_>(eta_bin) << "|"
                  << std::bitset<phi_bit_length_>(phi_bin);
        std::string binary_word = binary_ss.str();

        uint32_t packed_word = (et_bin  << (eta_bit_length_ + phi_bit_length_)) |
                               (eta_bin << phi_bit_length_) |
                               (phi_bin);

        if (idx == 0) {
            out << "Event : " << std::dec << iEvt << "\n";
        }
        out << "0x" << std::hex << std::setw(2) << std::setfill('0') << idx
            << " "  << binary_word
            << " 0x" << std::setw(8) << std::setfill('0') << packed_word << "\n";
        ++idx;
    }
}

void rootToTestVector(bool signalBool = true, bool vbfBool = true, unsigned int jzSlice = 3) {
    // --- Timing: event processing ---
    auto tStartProcessing = std::chrono::steady_clock::now();
    std::size_t totalEventsLooped = 0;   // total events we iterated over (across all files)

    // open input root
    TString inputFile = "/data/crystalwang/testVectorPipeline/testData/outputRootFiles/mc21_14TeV_hh_bbbb_vbf_novhh_e8557_s4422_r16130_DAOD_NTUPLE.root";
    std::cout << "opening file: " << inputFile << std::endl;

    TFile* f = TFile::Open(inputFile, "READ");
    if (!f || f->IsZombie()) {
        std::cerr << "Error: could not open " << inputFile << std::endl;
        return;
    }

    // Get the TTrees
    TTree* caloTopoTowerTree = (TTree*)f->Get("caloTopoTowerTree");
    TTree* topo422Tree       = (TTree*)f->Get("topo422Tree");
    TTree* gFexSRJTree       = (TTree*)f->Get("gFexSRJTree");
    TTree* jFexSRJTree       = (TTree*)f->Get("jFexSRJTree");

    if (!caloTopoTowerTree || !topo422Tree || !gFexSRJTree || !jFexSRJTree) {
        std::cerr << "Error: one or more TTrees not found in file!" << std::endl;
        f->Close();
        return;
    }

    // define the collTree structures and set branch addresses
    collTree calo, topo, gFex, jFex;
    branchify(caloTopoTowerTree, calo);
    branchify(topo422Tree, topo);
    branchify(gFexSRJTree, gFex);
    branchify(jFexSRJTree, jFex);

    OutputFiles out = makeMemPrintFilenames(signalBool, vbfBool, jzSlice);
    // open streams
    std::ofstream f_topotower(out.caloTopoTowers);
    std::ofstream f_topo(out.topo422);
    std::ofstream f_gfex(out.gFex);
    std::ofstream f_jfex(out.jFex);

    Long64_t nEntries = caloTopoTowerTree->GetEntries();

    for (Long64_t iEvt = 0; iEvt < nEntries; ++iEvt) {
        // extra index for measuring
        totalEventsLooped++;

        caloTopoTowerTree->GetEntry(iEvt);
        topo422Tree->GetEntry(iEvt);
        gFexSRJTree->GetEntry(iEvt);
        jFexSRJTree->GetEntry(iEvt);

        writeTestVectors(f_topotower, iEvt, calo);
        writeTestVectors(f_topo,      iEvt, topo);
        writeTestVectors(f_gfex,      iEvt, gFex);
        writeTestVectors(f_jfex,      iEvt, jFex);
    }
    
    // --- End of event processing timing ---
    auto tEndProcessing = std::chrono::steady_clock::now();
    std::chrono::duration<double> procTime = tEndProcessing - tStartProcessing;
    double timePerEvent = (totalEventsLooped > 0)
        ? procTime.count() / static_cast<double>(totalEventsLooped)
        : 0.0;

    std::cout << "\n[Timing] Event processing total: " << procTime.count() << " s\n"
              << "[Timing] Events processed (looped over): " << totalEventsLooped << "\n"
              << "[Timing] Avg time per event: " << timePerEvent << " s/event\n";

    f_topotower.close();
    f_topo.close();
    f_gfex.close();
    f_jfex.close();
    f->Close();
}
            
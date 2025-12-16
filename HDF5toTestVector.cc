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
#include "H5Cpp.h"
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
    static const std::string base = "/data/crystalwang/testVectorPipeline/testData/testVectorsHDF5";

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

// structure for the data from a single event 
struct singleEvent { 
    std::vector<double>* Et {nullptr}; 
    std::vector<double>* Eta {nullptr}; 
    std::vector<double>* Phi {nullptr}; 
};

// make a structure to store all the data and offsets from each collection
struct CollJagged {
    std::vector<double> Et;
    std::vector<double> Eta;
    std::vector<double> Phi;
    std::vector<int32_t> offsets;   // Awkward-style offsets

    size_t nEvents() const {
        if (offsets.size() < 2) {
            throw std::runtime_error("Offsets length < 2, cannot infer nEvents");
        }
        return offsets.size() - 1;  // standard Awkward convention
    }
};

// ditize and write test vectors
inline void writeTestVectors(std::ofstream& out, Long64_t iEvt, const singleEvent& v) {
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

// get the double data from the HDF5 file
std::vector<double> readDouble(H5::H5File& file, const std::string& path) {
    // open the corresponding dataset
    H5::DataSet dset = file.openDataSet(path);
    H5::DataSpace space = dset.getSpace();

    // confirm it's 1D
    int ndims = space.getSimpleExtentNdims();
    if (ndims != 1) {
        throw std::runtime_error("Dataset " + path + " is not 1D");
    }

    hsize_t dim;
    space.getSimpleExtentDims(&dim);
    std::vector<double> buf(dim);
    dset.read(buf.data(), H5::PredType::NATIVE_DOUBLE);
    return buf;
}

// get the int data from the hdf5 file
std::vector<int32_t> readInt(H5::H5File& file, const std::string& path) {
    H5::DataSet dset = file.openDataSet(path);
    H5::DataSpace space = dset.getSpace();

    int ndims = space.getSimpleExtentNdims();
    if (ndims != 1) {
        throw std::runtime_error("Dataset " + path + " is not 1D");
    }

    hsize_t dim;
    space.getSimpleExtentDims(&dim);
    std::vector<int32_t> buf(dim);
    dset.read(buf.data(), H5::PredType::NATIVE_INT32);
    return buf;
}

// Read a collection like "/CaloTopoTowers" or "/topo422"
CollJagged readCollection(H5::H5File& file, const std::string& groupName) {
    const std::string base = "/" + groupName;

    CollJagged c;
    c.Et   = readDouble(file, base + "/Et/data");
    c.Eta  = readDouble(file, base + "/Eta/data");
    c.Phi  = readDouble(file, base + "/Phi/data");
    c.offsets  = readInt(file, base + "/Et/offsets"); // shared offsets

    // sanity check
    if (c.Et.size() != c.Eta.size() ||
        c.Et.size() != c.Phi.size()) {
        throw std::runtime_error("Et/Eta/Phi data size mismatch in " + groupName);
    }
    return c;
}

// Build a singleEvent *for one event* using offsets
singleEvent makeEventColl(const CollJagged& c, size_t iEvt) {
    if (iEvt + 1 >= c.offsets.size()) {
        throw std::out_of_range("iEvt out of range in makeEventColl");
    }

    const int32_t start = c.offsets[iEvt];
    const int32_t stop  = c.offsets[iEvt + 1];
    if (start < 0 || stop < start || static_cast<size_t>(stop) > c.Et.size()) {
        throw std::runtime_error("Bad offsets for event in makeEventColl");
    }

    auto* Et  = new std::vector<double>(c.Et.begin()  + start, c.Et.begin()  + stop);
    auto* Eta = new std::vector<double>(c.Eta.begin() + start, c.Eta.begin() + stop);
    auto* Phi = new std::vector<double>(c.Phi.begin() + start, c.Phi.begin() + stop);

    singleEvent t;
    t.Et  = Et;
    t.Eta = Eta;
    t.Phi = Phi;
    return t;
}

void clearEvent(singleEvent& c) {
    delete c.Et;
    delete c.Eta;
    delete c.Phi;
    c.Et = c.Eta = c.Phi = nullptr;
}

void HDF5toTestVector(const std::string& h5filePath) {
    // --- Timing: event processing ---
    auto tStartProcessing = std::chrono::steady_clock::now();
    std::size_t totalEventsLooped = 0;   // total events we iterated over (across all files)
    
    // open file
    std::cout << "Opening HDF5 file: " << h5filePath << std::endl;
    H5::H5File file(h5filePath, H5F_ACC_RDONLY);

    // Read the four collections that correspond to your test-vector outputs
    CollJagged calo  = readCollection(file, "CaloTopoTowers");
    CollJagged topo  = readCollection(file, "topo422");
    CollJagged gfex  = readCollection(file, "gFEXSRJ");
    CollJagged jfex  = readCollection(file, "jFEXSRJ");

    // set the number of events
    size_t nEvt = calo.nEvents();
    if (topo.nEvents() != nEvt || gfex.nEvents() != nEvt || jfex.nEvents() != nEvt) {
        throw std::runtime_error("Collections have different nEvents in HDF5 file");
    }

    OutputFiles outNames = makeMemPrintFilenames(true, true, 3);

    std::ofstream f_calo(outNames.caloTopoTowers);
    std::ofstream f_topo(outNames.topo422);
    std::ofstream f_gfex(outNames.gFex);
    std::ofstream f_jfex(outNames.jFex);

    // for each event
    for (size_t iEvt = 0; iEvt < nEvt; ++iEvt) {

        totalEventsLooped++;

        // get the event of that index
        singleEvent caloEvt = makeEventColl(calo, iEvt);
        singleEvent topoEvt = makeEventColl(topo, iEvt);
        singleEvent gfexEvt = makeEventColl(gfex, iEvt);
        singleEvent jfexEvt = makeEventColl(jfex, iEvt);

        // write the test vectors
        writeTestVectors(f_calo, static_cast<Long64_t>(iEvt), caloEvt);
        writeTestVectors(f_topo, static_cast<Long64_t>(iEvt), topoEvt);
        writeTestVectors(f_gfex, static_cast<Long64_t>(iEvt), gfexEvt);
        writeTestVectors(f_jfex, static_cast<Long64_t>(iEvt), jfexEvt);

        // clear the data
        clearEvent(caloEvt);
        clearEvent(topoEvt);
        clearEvent(gfexEvt);
        clearEvent(jfexEvt);
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

}
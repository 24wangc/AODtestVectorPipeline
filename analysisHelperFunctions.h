#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>
#include <array>
//#include <ap_int.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "TH1F.h"
#include "TCanvas.h"
#include "TFile.h"
#include <cmath>
#include <TMath.h>
#include <unordered_map>
//#include "../algorithm/constants.h"

// ---------- Data container for one event ----------
struct EventDisplayInputs {
  TH2* heatmap = nullptr;  // e.g. sigTopo422Highest128SeedPositionsEvX
  std::vector<std::pair<double,double>> jFexSeeds;        // red, dashed
  std::vector<std::pair<double,double>> additionalSRJs;   // azure, dashed
  std::vector<std::pair<double,double>> newSeeds;         // green, solid
  std::vector<double>                  newSeedsEt;         // same size as newSeeds (for labels)
  TString outFile;                                         // output path
};

// Helper to create a TH2 with variable (or uniform) binning from the axes
static TH2F* MakeTH2Like(const char* name, const char* title,
                         const TAxis* axX, const TAxis* axY)
{
  const TArrayD& bx = *axX->GetXbins();
  const TArrayD& by = *axY->GetXbins();

  if (bx.GetSize() > 0 && by.GetSize() > 0)
    return new TH2F(name, title,
                    axX->GetNbins(), bx.GetArray(),
                    axY->GetNbins(), by.GetArray());
  if (bx.GetSize() > 0)
    return new TH2F(name, title,
                    axX->GetNbins(), bx.GetArray(),
                    axY->GetNbins(), axY->GetXmin(), axY->GetXmax());
  if (by.GetSize() > 0)
    return new TH2F(name, title,
                    axX->GetNbins(), axX->GetXmin(), axX->GetXmax(),
                    axY->GetNbins(), by.GetArray());
  return new TH2F(name, title,
                  axX->GetNbins(), axX->GetXmin(), axX->GetXmax(),
                  axY->GetNbins(), axY->GetXmin(), axY->GetXmax());
}



// Structure to store histograms used to generate background rate vs. signal efficiency plots
struct RateEffOut {
  TH1* hEff_vsThr;   // efficiency vs threshold (optional)
  TH1* hRate_vsThr;  // rate vs threshold in Hz (optional)
  TGraph* gRate_vsEff; // main output: y(rate) vs x(eff)
};

struct RateEff2DOut {
  TH2* hEff_vsThr_vsR;   // Signal efficiency surface vs (ET threshold, Rmax)
  TH2* hRate_vsThr_vsR;  // Background rate surface vs (ET threshold, Rmax)
  TGraphErrors* gRate_vsEff_all; // All (eff,rate) points from the 2D scan
};

// -----------------------------------------------------------------------------
// MakeRateVsEff_ScanRMax
// Scan ET threshold and Rmax simultaneously using 1D histograms.
// Inputs:
//   hSigEt, hBkgEt : 1D ET distributions (sig counts, bkg weighted to Hz)
//   hSigR,  hBkgR  : 1D R=psi_lead/psi_sublead distributions
// Returns 2D efficiency / rate surfaces and a graph with all (eff, rate) pairs.
// -----------------------------------------------------------------------------
RateEff2DOut MakeRateVsEff_ScanRMax(TH1* hSigEt, TH1* hBkgEt,
                                    TH1* hSigR,  TH1* hBkgR)
{
  // ---- (1) Cumulative along ET: pass if ET > threshold (HIGH -> LOW)
  auto* hSigCumEt = hSigEt->GetCumulative(/*forward=*/false,
                            (std::string(hSigEt->GetName())+"_cumul").c_str());
  auto* hBkgCumEt = hBkgEt->GetCumulative(/*forward=*/false,
                            (std::string(hBkgEt->GetName())+"_cumul").c_str());

  const int nbEt = hSigEt->GetNbinsX();
  const double totalSigEt = hSigEt->Integral(1, nbEt);

  std::vector<double> effEt(nbEt+1, 0.0), rateEt(nbEt+1, 0.0), errEt(nbEt+1, 0.0);
  for (int ib=1; ib<=nbEt; ++ib) {
    effEt[ib]  = (totalSigEt>0 ? hSigCumEt->GetBinContent(ib)/totalSigEt : 0.0);
    rateEt[ib] = hBkgCumEt->GetBinContent(ib);    // already in Hz if weighted
    errEt[ib]  = hBkgCumEt->GetBinError(ib);
  }

  // ---- (2) Cumulative along R: pass if R < Rmax (LOW -> HIGH)
  auto* hSigCumR_lo = hSigR->GetCumulative(/*forward=*/true,
                            (std::string(hSigR->GetName())+"_cumul_lo").c_str());
  auto* hBkgCumR_lo = hBkgR->GetCumulative(/*forward=*/true,
                            (std::string(hBkgR->GetName())+"_cumul_lo").c_str());

  const int nbR  = hSigR->GetNbinsX();
  const double totalSigR = hSigR->Integral(1, nbR);
  const double totalBkgR = hBkgR->Integral(1, nbR); // Hz if weighted

  std::vector<double> fracSigR(nbR+1, 0.0), fracBkgR(nbR+1, 0.0);
  for (int jb=1; jb<=nbR; ++jb) {
    fracSigR[jb] = (totalSigR>0 ? hSigCumR_lo->GetBinContent(jb)/totalSigR : 0.0);
    fracBkgR[jb] = (totalBkgR>0 ? hBkgCumR_lo->GetBinContent(jb)/totalBkgR : 0.0);
  }

  // ---- (3) Build 2D surfaces vs (ET_thr, Rmax)
  TH2F* hEff2D  = MakeTH2Like("hEff_vsThr_vsR",
                     ";Leading JetTagger LRJ E_{T} threshold [GeV];R_{max};Signal (hh#rightarrow4b) Efficiency",
                     hSigEt->GetXaxis(), hSigR->GetXaxis());
  TH2F* hRate2D = MakeTH2Like("hRate_vsThr_vsR",
                     ";Leading JetTagger LRJ E_{T} threshold [GeV];R_{max};Estimated Background Rate [Hz]",
                     hBkgEt->GetXaxis(), hBkgR->GetXaxis());

  for (int ib=1; ib<=nbEt; ++ib) {
    for (int jb=1; jb<=nbR; ++jb) {
      const double eff  = std::clamp(effEt[ib]  * fracSigR[jb], 0.0, 1.0);
      const double rate = rateEt[ib] * fracBkgR[jb];
      hEff2D ->SetBinContent(ib, jb, eff);
      hRate2D->SetBinContent(ib, jb, rate);
      // (Optional) error on rate:
      hRate2D->SetBinError(ib, jb, errEt[ib] * fracBkgR[jb]);
    }
  }

  // ---- (4) Collect all (eff, rate) points into a TGraphErrors
  const int npts = nbEt * nbR;
  auto* g = new TGraphErrors(npts);
  g->SetName("gRate_vsEff_all");
  g->SetTitle("Trigger Rate vs Signal Efficiency;Signal (hh#rightarrow4b) Efficiency;Estimated Background Rate [Hz]");

  int k=0;
  for (int ib=1; ib<=nbEt; ++ib) {
    for (int jb=1; jb<=nbR; ++jb, ++k) {
      const double eff  = hEff2D ->GetBinContent(ib,jb);
      const double rate = hRate2D->GetBinContent(ib,jb);
      const double err  = hRate2D->GetBinError  (ib,jb);
      g->SetPoint(k, eff, rate);
      g->SetPointError(k, 0.0, err);
    }
  }

  return {hEff2D, hRate2D, g};
}

// Function to generate background rate vs. signal efficiency plots
RateEffOut MakeRateVsEff(TH1* hSig, TH1* hBkg) {
  // 1) cumulative from HIGH -> LOW threshold
  auto* hSigCum = hSig->GetCumulative(/*forward=*/false, (std::string(hSig->GetName())+"_cumul").c_str());
  auto* hBkgCum = hBkg->GetCumulative(/*forward=*/false, (std::string(hBkg->GetName())+"_cumul").c_str());

  // 2) signal efficiency vs threshold
  const int nb = hSig->GetNbinsX();
  const double totalSig = hSig->Integral(1, nb);  // unweighted total signal events
  auto* hEff = (TH1*)hSigCum->Clone((std::string(hSig->GetName())+"_eff_vs_thr").c_str());
  hEff->SetTitle(";Leading JetTagger LRJ E_{T} threshold [GeV]; Signal (hh->4b) Efficiency");
  for (int ib = 1; ib <= nb; ++ib) {
    double eff = (totalSig > 0 ? hSigCum->GetBinContent(ib)/totalSig : 0.0);
    hEff->SetBinContent(ib, std::clamp(eff, 0.0, 1.0));
    hEff->SetBinError  (ib, 0.0); // (optional: compute binomial errors if you want)
  }

  // 3) background rate vs threshold (already in Hz because you weighted on fill)
  auto* hRate = (TH1*)hBkgCum->Clone((std::string(hBkg->GetName())+"_rate_vs_thr").c_str());
  hRate->SetTitle(";Leading JetTagger LRJ E_{T} threshold [GeV];Estimated Background Rate [Hz]");

  // 4) build Rate vs Efficiency curve (one point per threshold bin)
  auto* g = new TGraphErrors(nb);
  g->SetName("gRate_vsEff");
  g->SetTitle("Trigger Rate vs Signal Efficiency; Signal (hh->4b) Efficiency;Estimated Background Rate [Hz]");
  for (int ib = 1; ib <= nb; ++ib) {
    const double eff  = hEff ->GetBinContent(ib);
    const double rate = hRate->GetBinContent(ib);
    double xerr = 0.0;              // no threshold uncertainty here
    double yerr = hRate->GetBinError(ib);  // background stat. error
    g->SetPoint(ib-1, eff, rate);
    g->SetPointError(ib-1, xerr, yerr);
  }

  return {hEff, hRate, g};
}

double FindThrForRate(const TH1* hRateVsThr, double targetHz) {
  const int n = hRateVsThr->GetNbinsX();
  double x0 = hRateVsThr->GetXaxis()->GetBinCenter(1);
  double y0 = hRateVsThr->GetBinContent(1);

  for (int i = 2; i <= n; ++i) {
    double x1 = hRateVsThr->GetXaxis()->GetBinCenter(i);
    double y1 = hRateVsThr->GetBinContent(i);

    // look for a bracket around targetHz (works if rate is monotonic vs threshold)
    const bool brackets = ( (y0 - targetHz)*(y1 - targetHz) <= 0 );
    if (brackets && y1 != y0) {
      const double t = (targetHz - y0) / (y1 - y0);    // linear interpolation fraction
      return x0 + t * (x1 - x0);                       // threshold (GeV)
    }
    x0 = x1; y0 = y1;
  }
  return std::numeric_limits<double>::quiet_NaN();     // not found
}


// ---------- Draw ONE event to a canvas and save ----------
void DrawEventDisplay(const EventDisplayInputs& ev,
                      TCanvas& c,
                      double circleR = 1.0,
                      double labelDy  = 0.6) {
  if (!ev.heatmap) return;

  c.Clear();   // reuse the same canvas safely
  c.cd();

  // Heatmap
  ev.heatmap->GetZaxis()->SetTitle("E_{T} [GeV]");
  ev.heatmap->Draw("COLZ");

  // jFEX seeds: red, dashed
  for (const auto& p : ev.jFexSeeds) {
    TEllipse* e = new TEllipse(p.first, p.second, circleR, circleR);
    e->SetLineColor(kRed);
    e->SetLineWidth(2);
    e->SetFillStyle(0);
    e->SetLineStyle(2);   // dashed
    e->Draw("same");
  }

  // Additional SRJs: azure, dashed
  for (const auto& p : ev.additionalSRJs) {
    TEllipse* e = new TEllipse(p.first, p.second, circleR, circleR);
    e->SetLineColor(kAzure+2);
    e->SetLineWidth(2);
    e->SetFillStyle(0);
    e->SetLineStyle(2);   // dashed
    e->Draw("same");
  }

  // New seeds: green, solid + labels
  const size_t nlab = std::min(ev.newSeeds.size(), ev.newSeedsEt.size());
  for (size_t i = 0; i < ev.newSeeds.size(); ++i) {
    const auto& p = ev.newSeeds[i];
    TEllipse* e = new TEllipse(p.first, p.second, circleR, circleR);
    e->SetLineColor(kGreen+2);
    e->SetLineWidth(2);
    e->SetFillStyle(0);
    e->SetLineStyle(1);   // solid
    e->Draw("same");

    if (i < nlab) {
      TString lab = Form("#eta=%.2f, #phi=%.2f, E_{T}=%.1f GeV", p.first, p.second, ev.newSeedsEt[i]);
      TLatex lat;
      lat.SetTextSize(0.025);
      lat.SetTextColor(kBlack);
      lat.DrawLatex(p.first, p.second + labelDy, lab);
    }
  }

  c.Modified(); c.Update();
  if (!ev.outFile.IsNull()) c.SaveAs(ev.outFile);
}

// ---------- Draw MANY events (loop) ----------
void DrawManyEventDisplays(const std::vector<EventDisplayInputs>& events,
                           TCanvas& c,
                           double circleR = 1.0,
                           double labelDy  = 0.6) {
  for (const auto& ev : events) {
    DrawEventDisplay(ev, c, circleR, labelDy);
  }
}


// ---------- Standalone top ROC (no subplot), same styling, with legend + x-axis ----------
void SaveStandaloneROC(const char* cname,
                       const std::vector<TGraph*>& graphs,
                       TLegend* srcLegend,
                       const TString& outpath,
                       double ymin = 1.0, double ymax = 1e4) {
    if (graphs.empty() || !graphs[0]) return;

    TCanvas* c = new TCanvas(cname, cname, 700, 650);
    c->SetLogy();
    c->SetTicks(1, 1);

    // Margins: give more space for labels/titles
    c->SetLeftMargin(0.16);
    c->SetBottomMargin(0.16);
    c->SetRightMargin(0.05);
    c->SetTopMargin(0.05);

    // First graph
    TGraph* g0 = (TGraph*)graphs[0]->Clone(TString(cname) + "_g0");
    g0->SetTitle("ROC Curve;Signal Efficiency;Background Rejection (1/FPR)");
    g0->GetXaxis()->SetRangeUser(0.0, 1.0);
    //g0->GetYaxis()->SetRangeUser(ymin, ymax);

    // Axis label/title sizes
    g0->GetXaxis()->SetLabelSize(0.045);
    g0->GetXaxis()->SetTitleSize(0.055);
    g0->GetYaxis()->SetLabelSize(0.045);
    g0->GetYaxis()->SetTitleSize(0.055);
    g0->GetYaxis()->CenterTitle(true);

    g0->Draw("AL");

    for (size_t i = 1; i < graphs.size(); ++i) {
        if (!graphs[i]) continue;
        TGraph* gi = (TGraph*)graphs[i]->Clone(TString(cname) + Form("_g%zu", i));
        gi->Draw("L SAME");
    }

    // Legend: smaller, moved slightly down/left
    if (srcLegend) {
        TLegend* leg = (TLegend*)srcLegend->Clone(TString(cname) + "_legend");
        leg->SetX1NDC(0.58); leg->SetY1NDC(0.58);
        leg->SetX2NDC(0.88); leg->SetY2NDC(0.83);
        leg->SetTextSize(0.032);
        leg->SetBorderSize(0);   // cleaner look
        leg->SetFillStyle(0);    // transparent
        leg->Draw();
    }

    c->Modified(); c->Update();
    c->SaveAs(outpath);
}

// ------------------------------------------------------------
// Style + draw a stack of 10 histograms onto one canvas and
// save as a single-page PDF.
// ------------------------------------------------------------
void OverlayAndSave(TH1F* hists[], int n, const char* canvasName,
                    TString pdfOut,
                    const char* drawOpt = "HIST")
{
    if (n <= 0 || !hists[0]) return;

    // Color palette to mimic your legend image:
    // JZ0 red, JZ1 green, JZ2 blue, JZ3 yellow, JZ4 magenta,
    // JZ5 cyan, JZ6 teal-ish, JZ7 violet, JZ8 light gray, JZ9 dark gray.
    const Color_t colors[10] = {
        kRed+1,      // JZ0
        kGreen+1,    // JZ1
        kBlue,       // JZ2
        kYellow+1,   // JZ3
        kMagenta+1,  // JZ4
        kCyan+1,     // JZ5
        kTeal+2,     // JZ6
        kViolet+1,   // JZ7
        kGray+1,     // JZ8
        kGray+3      // JZ9
    };

    TCanvas* c = new TCanvas(canvasName, canvasName, 900, 700);
    c->SetMargin(0.12, 0.22, 0.16, 0.06); // leave room for right-side legend
    c->SetLogy();

    // Build legend on the right (similar look to your screenshot)
    // (colored boxes, no border, transparent background)
    TLegend* leg = new TLegend(0.80, 0.15, 0.97, 0.92);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.030);

    // Determine a nice max so everything fits
    //double ymax = 0.0;

    for (int i = 0; i < n; ++i) {
        if (!hists[i]) continue;
        // Style
        const Color_t col = colors[i % 10];
        hists[i]->SetLineColor(col);
        hists[i]->SetLineWidth(2);
        hists[i]->SetFillColor(col);
        hists[i]->SetFillStyle(0); // solid box in legend
        hists[i]->GetXaxis()->SetTitleSize(0.045);
        hists[i]->GetYaxis()->SetTitleSize(0.045);
        hists[i]->GetXaxis()->SetLabelSize(0.040);
        hists[i]->GetYaxis()->SetLabelSize(0.040);
        hists[i]->SetMinimum(1e-7);
        hists[i]->SetMaximum(5e8);

        // Track max
        //if (hists[i]->GetMaximum() > ymax) ymax = hists[i]->GetMaximum();
    }

    // Draw first one to set axes
    //hists[0]->SetMaximum(ymax * 1.20);
    hists[n - 1]->Draw("HIST");

    // Draw the rest
    for (int i = n - 2; i >= 0; i--) {
        if (!hists[i]) continue;
        hists[i]->Draw(TString::Format("HIST SAME"));
    }

    // Add legend entries in order JZ0..JZ9 with colored boxes
    for (int i = 0; i < n; ++i) {
        if (!hists[i]) continue;
        const Color_t col = colors[i % 10];
        leg->AddEntry(hists[i], TString::Format("Slice JZ%d", i), "f")->SetFillColor(col);
    }
    leg->Draw();

    // after drawing all hists
    double x1 = gPad->GetUxmin(), x2 = gPad->GetUxmax();
    double y1 = 1e-7, y2 = 1e-7 * pow(10, 0.03);  // a very thin band above min
    TBox* cover = new TBox(x1, y1, x2, y2);
    cover->SetFillColor(kWhite);
    cover->SetLineColor(kWhite);
    cover->Draw("same");
    gPad->RedrawAxis();

    // Write the PDF
    c->SaveAs(pdfOut);

    // (optional) keep objects alive outside this function if needed
    // delete c; // uncomment if you manage object ownership elsewhere
}

constexpr unsigned int nJZSlices_ = 10;


// Constants for Event Display Generation
constexpr unsigned int evDisplayMinSig_ = 15;
constexpr unsigned int evDisplay0Sig_ = evDisplayMinSig_ + 1;
constexpr unsigned int evDisplay1Sig_ = evDisplay0Sig_ + 1;
constexpr unsigned int evDisplay2Sig_ = evDisplay1Sig_ + 1;
constexpr unsigned int evDisplay3Sig_ = evDisplay2Sig_ + 1;
constexpr unsigned int evDisplay4Sig_ = evDisplay3Sig_ + 1;
constexpr unsigned int evDisplayMaxSig_ = evDisplay4Sig_;

constexpr unsigned int evDisplayMinBack_ = 10;
constexpr unsigned int evDisplay0Back_ = evDisplayMinBack_ + 1;
constexpr unsigned int evDisplay1Back_ = evDisplay0Back_ + 1;
constexpr unsigned int evDisplay2Back_ = evDisplay1Back_ + 1;
constexpr unsigned int evDisplay3Back_ = evDisplay2Back_ + 1;
constexpr unsigned int evDisplay4Back_ = evDisplay3Back_ + 1;
constexpr unsigned int evDisplayMaxBack_ = evDisplay4Back_;

bool displayEv0Sig_ = false;
bool displayEv1Sig_ = false;
bool displayEv2Sig_ = false;
bool displayEv3Sig_ = false;
bool displayEv4Sig_ = false;

bool displayEv0Back_ = false;
bool displayEv1Back_ = false;
bool displayEv2Back_ = false;
bool displayEv3Back_ = false;
bool displayEv4Back_ = false;

const double mH_ = 125.0;

constexpr double et_granularity_ = 0.125;
constexpr unsigned int et_bit_length_ = 13;
constexpr unsigned int eta_bit_length_ = 8;
constexpr unsigned int phi_bit_length_ = 6;
constexpr unsigned int psi_R_bit_length_ = 10; 
constexpr unsigned int padded_zeroes_length_ = 64 - et_bit_length_ - phi_bit_length_ - psi_R_bit_length_;
constexpr double phi_min_ = -3.2;
constexpr double phi_max_ = 3.2;
constexpr double eta_min_ = -5.0;
constexpr double eta_max_ = 5.0;
constexpr double eta_granularity_ = 0.0390625;
constexpr double phi_granularity_ = 0.1;
constexpr unsigned int et_min_ = 0;
constexpr unsigned int et_max_ = 1024;
const bool useMax_ = false;
const unsigned nSeeds_ = 2;

inline double undigitize_phi(const std::bitset<phi_bit_length_>& phi_bits) {
    return phi_min_ + phi_bits.to_ulong() * (6.4 / 256.0);
}

inline double undigitize_eta(const std::bitset<eta_bit_length_>& eta_bits) {
    return eta_min_ + eta_bits.to_ulong() * (10.0 / 2048.0);
}

inline double undigitize_et(const std::bitset<et_bit_length_>& et_bits) {
    return et_bits.to_ulong() * et_granularity_;
}

inline double undigitize_nmio(const std::bitset<psi_R_bit_length_>& nmio_bits) {
    return nmio_bits.to_ulong() * 0.0048828125; // 5 / 2^psi_R_bit_length_
}

// Function to scale and digitize a value, returning the result as a binary string
template <int bit_length>
inline std::bitset<bit_length > digitize(double value, double min_val, double max_val) {
    // Ensure the value is within range
    if (value < min_val || value > max_val) {
        std::cerr << "Value is out of range" << "\n";
    }
    //std::cout << "value : " << value << "\n";

    // Calculate scale factor
    double scale = (std::pow(2, bit_length)) / (max_val - min_val);

    //std::cout << "scale: " << scale << "\n";
    
    // Calculate the digitized value and round it
    int digitized_value = static_cast<int>(std::round((value - min_val) * scale));

    //std::cout << "digitized value: " << digitized_value << "\n";
    
    // Return the digitized value as a binary string (with leading zeros to fit the bit length)
    std::bitset<bit_length > bin(digitized_value);
    //std::pair<std::bitset<bit_length >, unsigned int> string_dig_pair(bin, digitized_value);
    return bin; // Extract only the relevant bits
}

double calcDeltaR2(double eta1, double phi1, double eta2, double phi2) {
    //std::cout << "eta1: " << eta1 << " eta2: " << eta2 << " phi1: " << phi1 << " phi2: " << phi2 << "\n";
    double dEta = eta1 - eta2;
    double dPhi = std::abs(phi1 - phi2);
    
    // Ensure Δφ is within [-π, π] range
    if (dPhi > M_PI) {
        dPhi = 2 * M_PI - dPhi;
    }
    //std::cout << "dr2: " << dEta * dEta + dPhi * dPhi << "\n";
    return dEta * dEta + dPhi * dPhi;
}

bool energyMax(double et, double et_cutoff){
    return et <= et_cutoff;
}

bool energyMin(double et, double et_cutoff){
    return et >= et_cutoff;
}

inline void redirect_output_to_file() {
    // Open the file in write mode
    freopen("output.txt", "w", stdout);  // This will redirect stdout to output.txt
}

void SetPlotStyle();
void mySmallText(Double_t x, Double_t y, Color_t color, char* text);

void SetPlotStyle() {
    // from ATLAS plot style macro

    // use plain black on white colors
    gStyle->SetFrameBorderMode(0);
    gStyle->SetFrameFillColor(0);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetCanvasColor(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadColor(0);
    gStyle->SetStatColor(0);
    gStyle->SetHistLineColor(1);

    gStyle->SetPalette(1);

    // set the paper & margin sizes
    gStyle->SetPaperSize(20, 26);
    gStyle->SetPadTopMargin(0.05);
    gStyle->SetPadRightMargin(0.15);
    gStyle->SetPadBottomMargin(0.16);
    gStyle->SetPadLeftMargin(0.16);

    // set title offsets (for axis label)
    gStyle->SetTitleXOffset(1.4);
    gStyle->SetTitleYOffset(1.4);

    // use large fonts
    gStyle->SetTextFont(42);
    gStyle->SetTextSize(0.07);
    gStyle->SetLabelFont(42, "x");
    gStyle->SetTitleFont(42, "x");
    gStyle->SetLabelFont(42, "y");
    gStyle->SetTitleFont(42, "y");
    gStyle->SetLabelFont(42, "z");
    gStyle->SetTitleFont(42, "z");
    gStyle->SetLabelSize(0.05, "x");
    gStyle->SetTitleSize(0.05, "x");
    gStyle->SetLabelSize(0.05, "y");
    gStyle->SetTitleSize(0.05, "y");
    gStyle->SetLabelSize(0.05, "z");
    gStyle->SetTitleSize(0.05, "z");

    // use bold lines and markers
    gStyle->SetMarkerStyle(20);
    gStyle->SetMarkerSize(1.2);
    gStyle->SetHistLineWidth(2.);
    gStyle->SetLineStyleString(2, "[12 12]");

    // get rid of error bar caps
    gStyle->SetEndErrorSize(0.);

    // do not display any of the standard histogram decorations
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);

    // put tick marks on top and RHS of plots
    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);
}

void mySmallText(Double_t x, Double_t y, Color_t color, char* text) {
    Double_t tsize = 0.044;
    TLatex l;
    l.SetTextSize(tsize);
    l.SetNDC();
    l.SetTextColor(color);
    l.DrawLatex(x, y, text);
}
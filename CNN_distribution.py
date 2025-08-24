#This is to plot the distribution of the CNN. 
import ROOT
import numpy as np

ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)

# --- Load ROOT file and tree ---
rdf = ROOT.RDataFrame("PredictionTree", "cnn_predictions_ak15_3channel_weighted_signalup.root")

# --- Filter signal and background separately ---
rdf_sig = rdf.Filter("label == 1")
rdf_bkg = rdf.Filter("label == 0")

# --- Count total events for normalization ---
n_sig = rdf_sig.Count().GetValue()
n_bkg = rdf_bkg.Count().GetValue()

# --- Define normalized weights (percentage) ---
rdf_sig = rdf_sig.Define("norm_weight", f"1.0 / {n_sig}")
rdf_bkg = rdf_bkg.Define("norm_weight", f"1.0 / {n_bkg}")

# --- Book histograms ---
bins = (50, 0.0, 1.0)
hist_sig = rdf_sig.Histo1D(("sig_hist", "CNN Score;NN Score;Percentage of Signal Events", *bins), "score_8b", "norm_weight")
hist_bkg = rdf_bkg.Histo1D(("bkg_hist", "CNN Score;NN Score;Percentage of Background Events", *bins), "score_8b", "norm_weight")

# --- Style ---
hist_sig.SetLineColor(ROOT.kBlue)
hist_sig.SetLineWidth(2)
hist_sig.SetTitle("CNN Score Distribution (Normalized by Class)")

hist_bkg.SetLineColor(ROOT.kRed)
hist_bkg.SetLineWidth(2)

# --- Scale to percent (Y-axis as percent, still integrates to 1 per class) ---
hist_sig.Scale(100, "width")
hist_bkg.Scale(100, "width")

# --- Draw ---
canvas = ROOT.TCanvas("c", "", 800, 600)

hist_bkg.Draw("HIST")
hist_sig.Draw("HIST SAME")

legend = ROOT.TLegend(0.6, 0.75, 0.88, 0.88)
legend.AddEntry(hist_sig.GetPtr(), "Signal", "l")
legend.AddEntry(hist_bkg.GetPtr(), "Background", "l")
legend.Draw()

canvas.SetGrid()
canvas.SaveAs("score_distribution_3channel_weighted_signalup.root.png")

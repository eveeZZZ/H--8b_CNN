#This is to print an average image of AK8 jets for a particular file. 
"""
import ROOT
import numpy as np
import math
from DataFormats.FWLite import Events, Handle

ROOT.FWLiteEnabler.enable()

# Input MiniAOD file
input_file = "/hdfs/store/user/abdollah/ExoHiggs_WithAK15/MiniAOD/WJet_HT400_600/OUTROOT_WJet_HT400_600_11_Mini.root"
events = Events(f"file:{input_file}")

# Handles
ak8_handle = Handle("std::vector<pat::Jet>")
ak8_label = ("slimmedJetsAK8", "", "PAT")

pf_handle = Handle("std::vector<pat::PackedCandidate>")
pf_label = ("packedPFCandidates", "", "PAT")

# Histogram parameters
grid_size = 10
half_range = 0.5
bin_width = (2 * half_range) / grid_size

# Create two 2D histograms: raw pT and normalized
hist_raw = ROOT.TH2F("jet_image_raw", "Jet Image (Raw pT);#Delta#eta;#Delta#phi", grid_size, -0.5, 0.5, grid_size, -0.5, 0.5)
hist_norm = ROOT.TH2F("jet_image_norm", "Jet Image (Normalized);#Delta#eta;#Delta#phi", grid_size, -0.5, 0.5, grid_size, -0.5, 0.5)

# Loop over events
for i, event in enumerate(events):
    if i >= 200:
        break

    event.getByLabel(ak8_label, ak8_handle)
    jets = ak8_handle.product()
    if jets.size() == 0:
        continue

    lead_jet = jets.at(0)
    jet_eta, jet_phi = lead_jet.eta(), lead_jet.phi()

    event.getByLabel(pf_label, pf_handle)
    pf_cands = pf_handle.product()

    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    total_pt = 0.0

    for cand in pf_cands:
        deta = cand.eta() - jet_eta
        dphi = math.atan2(math.sin(cand.phi() - jet_phi), math.cos(cand.phi() - jet_phi))

        if abs(deta) > half_range or abs(dphi) > half_range:
            continue

        eta_bin = int((deta + half_range) / bin_width)
        phi_bin = int((dphi + half_range) / bin_width)

        if 0 <= eta_bin < grid_size and 0 <= phi_bin < grid_size:
            grid[eta_bin, phi_bin] += cand.pt()
            total_pt += cand.pt()

    # Fill raw and normalized histograms
    for eta_bin in range(grid_size):
        for phi_bin in range(grid_size):
            deta_center = -0.5 + (eta_bin + 0.5) * bin_width
            dphi_center = -0.5 + (phi_bin + 0.5) * bin_width
            pt_val = grid[eta_bin, phi_bin]
            hist_raw.Fill(deta_center, dphi_center, pt_val)
            if total_pt > 0:
                hist_norm.Fill(deta_center, dphi_center, pt_val / total_pt)

# Plotting
canvas = ROOT.TCanvas("canvas", "", 1000, 500)
canvas.Divide(2, 1)

canvas.cd(1)
ROOT.gPad.SetRightMargin(0.15)
hist_raw.Draw("COLZ TEXT")

canvas.cd(2)
ROOT.gPad.SetRightMargin(0.15)
hist_norm.Draw("COLZ TEXT")

canvas.SaveAs("jet_image_comparison.png")
print("✅ Saved 2D jet image histograms to jet_image_comparison.png")
"""
import ROOT
import numpy as np
import math
from DataFormats.FWLite import Events, Handle

ROOT.FWLiteEnabler.enable()

# Input MiniAOD file
#input_file = "/hdfs/store/user/abdollah/ExoHiggs_WithAK15/MiniAOD/8bjets/OUTROOT_8bjets_59_Mini.root"
#input_file = "/hdfs/store/user/abdollah/ExoHiggs_WithAK15/MiniAOD/WJet_HT600_800/OUTROOT_WJet_HT600_800_426_Mini.root"
input_file = "/hdfs/store/user/abdollah/ExoHiggs_WithAK15/MiniAOD/4bjets/OUTROOT_4bjets_15_Mini.root"



events = Events(f"file:{input_file}")

# Handles
ak8_handle = Handle("std::vector<pat::Jet>")
ak8_label = ("slimmedJetsAK8", "", "PAT")

pf_handle = Handle("std::vector<pat::PackedCandidate>")
pf_label = ("packedPFCandidates", "", "PAT")

# Histogram parameters
grid_size = 10
half_range = 0.5
bin_width = (2 * half_range) / grid_size
floor_value = 1e-5  # small non-zero to keep empty bins visible

# Create two 2D histograms: raw pT and normalized
hist_raw = ROOT.TH2F("jet_image_raw", "Jet Image (Raw pT);#Delta#eta;#Delta#phi", grid_size, -0.5, 0.5, grid_size, -0.5, 0.5)
hist_norm = ROOT.TH2F("jet_image_norm", "Jet Image (Normalized);#Delta#eta;#Delta#phi", grid_size, -0.5, 0.5, grid_size, -0.5, 0.5)

# Loop over events
for i, event in enumerate(events):
    if i >= 200:
        break

    event.getByLabel(ak8_label, ak8_handle)
    jets = ak8_handle.product()
    if jets.size() == 0:
        continue

    lead_jet = jets.at(0)
    if lead_jet.pt() < 300:
        continue  # skip jets with pt < 300 GeV

    jet_eta, jet_phi = lead_jet.eta(), lead_jet.phi()

    event.getByLabel(pf_label, pf_handle)
    pf_cands = pf_handle.product()

    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    total_pt = 0.0

    for cand in pf_cands:
        deta = cand.eta() - jet_eta
        dphi = math.atan2(math.sin(cand.phi() - jet_phi), math.cos(cand.phi() - jet_phi))

        if abs(deta) > half_range or abs(dphi) > half_range:
            continue

        eta_bin = int((deta + half_range) / bin_width)
        phi_bin = int((dphi + half_range) / bin_width)

        if 0 <= eta_bin < grid_size and 0 <= phi_bin < grid_size:
            grid[eta_bin, phi_bin] += cand.pt()
            total_pt += cand.pt()

    # Fill histograms, apply floor to zero bins
    for eta_bin in range(grid_size):
        for phi_bin in range(grid_size):
            deta_center = -0.5 + (eta_bin + 0.5) * bin_width
            dphi_center = -0.5 + (phi_bin + 0.5) * bin_width
            pt_val = grid[eta_bin, phi_bin]

            # raw
            hist_raw.Fill(deta_center, dphi_center, pt_val if pt_val > 0 else floor_value)

            # normalized
            if total_pt > 0:
                norm_val = pt_val / total_pt
                hist_norm.Fill(deta_center, dphi_center, norm_val if norm_val > 0 else floor_value)
            else:
                hist_norm.Fill(deta_center, dphi_center, floor_value)

# Plotting
canvas = ROOT.TCanvas("canvas", "", 1000, 500)
canvas.Divide(2, 1)

canvas.cd(1)
ROOT.gPad.SetRightMargin(0.15)
hist_raw.Draw("COLZ TEXT")

canvas.cd(2)
ROOT.gPad.SetRightMargin(0.15)
hist_norm.Draw("COLZ TEXT")

canvas.SaveAs("jet_image_comparison.png")
print("✅ Saved 2D jet image histograms to jet_image_comparison.png")


"""
import ROOT
import numpy as np
import math
from DataFormats.FWLite import Events, Handle

ROOT.FWLiteEnabler.enable()

# Input file
input_file = "/hdfs/store/user/abdollah/ExoHiggs_WithAK15/MiniAOD/4bjets/OUTROOT_4bjets_10_Mini.root"
events = Events(f"file:{input_file}")

# Handles
ak8_handle = Handle("std::vector<pat::Jet>")
ak8_label = ("slimmedJetsAK8", "", "PAT")

pf_handle = Handle("std::vector<pat::PackedCandidate>")
pf_label = ("packedPFCandidates", "", "PAT")

# Image parameters
grid_size = 20
half_range = 0.5
bin_width = (2 * half_range) / grid_size
epsilon = 1e-6  # for safe division
floor_value = 1e-5  # for visible bin content

# Process the first valid event
for i, event in enumerate(events):
    event.getByLabel(ak8_label, ak8_handle)
    jets = ak8_handle.product()
    if jets.size() == 0:
        continue

    lead_jet = jets.at(0)
    jet_eta, jet_phi = lead_jet.eta(), lead_jet.phi()

    event.getByLabel(pf_label, pf_handle)
    pf_cands = pf_handle.product()

    # Initialize image grid
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    total_pt = 0.0

    for cand in pf_cands:
        deta = cand.eta() - jet_eta
        dphi = math.atan2(math.sin(cand.phi() - jet_phi), math.cos(cand.phi() - jet_phi))
        if abs(deta) > half_range or abs(dphi) > half_range:
            continue

        eta_bin = int((deta + half_range) / bin_width)
        phi_bin = int((dphi + half_range) / bin_width)
        if 0 <= eta_bin < grid_size and 0 <= phi_bin < grid_size:
            grid[eta_bin, phi_bin] += cand.pt()
            total_pt += cand.pt()

    # Create histograms
    hist_raw = ROOT.TH2F("jet_image_raw", "Jet Image (Raw pT);#Delta#eta;#Delta#phi",
                         grid_size, -0.5, 0.5, grid_size, -0.5, 0.5)
    hist_norm = ROOT.TH2F("jet_image_norm", "Jet Image (Normalized pT);#Delta#eta;#Delta#phi",
                          grid_size, -0.5, 0.5, grid_size, -0.5, 0.5)

    safe_total_pt = max(total_pt, epsilon)

    # Fill histograms with floor offset for zeros
    for eta_bin in range(grid_size):
        for phi_bin in range(grid_size):
            deta_center = -0.5 + (eta_bin + 0.5) * bin_width
            dphi_center = -0.5 + (phi_bin + 0.5) * bin_width
            pt_val = grid[eta_bin, phi_bin]

            # Fill raw pT histogram
            fill_raw = pt_val if pt_val > 0 else floor_value
            hist_raw.Fill(deta_center, dphi_center, fill_raw)

            # Fill normalized histogram
            norm_val = pt_val / safe_total_pt
            fill_norm = norm_val if norm_val > 0 else floor_value
            hist_norm.Fill(deta_center, dphi_center, fill_norm)

    print(f"✅ Processed event {i}")
    break  # Only one event

# Style
ROOT.gStyle.SetNumberContours(100)
ROOT.gStyle.SetPalette(ROOT.kViridis)

# Plotting
canvas = ROOT.TCanvas("canvas", "", 1200, 600)
canvas.Divide(2, 1)

canvas.cd(1)
ROOT.gPad.SetRightMargin(0.15)
hist_raw.Draw("COLZ TEXT")

canvas.cd(2)
ROOT.gPad.SetRightMargin(0.15)
hist_norm.Draw("COLZ TEXT")

canvas.SaveAs("jet_image_single_event.png")
print("✅ Saved: jet_image_single_event.png")

"""
"""
import ROOT
import numpy as np
import math
from DataFormats.FWLite import Events, Handle

ROOT.FWLiteEnabler.enable()

# Input file
input_file = "/hdfs/store/user/abdollah/ExoHiggs_WithAK15/MiniAOD/4bjets/OUTROOT_4bjets_10_Mini.root"
events = Events(f"file:{input_file}")

# Handles
ak8_handle = Handle("std::vector<pat::Jet>")
ak8_label = ("slimmedJetsAK8", "", "PAT")

pf_handle = Handle("std::vector<pat::PackedCandidate>")
pf_label = ("packedPFCandidates", "", "PAT")

# Image parameters
grid_size = 10
half_range = 0.5  # Δη and Δφ ∈ [-0.5, 0.5]
bin_width = (2 * half_range) / grid_size  # = 0.1
epsilon = 1e-6  # for safe division
floor_value = 1e-5  # to show empty bins

# Process the first valid event
for i, event in enumerate(events):
    event.getByLabel(ak8_label, ak8_handle)
    jets = ak8_handle.product()
    if jets.size() == 0:
        continue

    lead_jet = jets.at(0)
    jet_eta, jet_phi = lead_jet.eta(), lead_jet.phi()

    event.getByLabel(pf_label, pf_handle)
    pf_cands = pf_handle.product()

    # Initialize image grid
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    total_pt = 0.0

    for cand in pf_cands:
        deta = cand.eta() - jet_eta
        dphi = math.atan2(math.sin(cand.phi() - jet_phi), math.cos(cand.phi() - jet_phi))
        if abs(deta) > half_range or abs(dphi) > half_range:
            continue

        eta_bin = int((deta + half_range) / bin_width)
        phi_bin = int((dphi + half_range) / bin_width)
        if 0 <= eta_bin < grid_size and 0 <= phi_bin < grid_size:
            grid[eta_bin, phi_bin] += cand.pt()
            total_pt += cand.pt()

    # Create histograms
    hist_raw = ROOT.TH2F("jet_image_raw", "Jet Image (Raw pT);#Delta#eta;#Delta#phi",
                         grid_size, -half_range, half_range, grid_size, -half_range, half_range)
    hist_norm = ROOT.TH2F("jet_image_norm", "Jet Image (Normalized pT);#Delta#eta;#Delta#phi",
                          grid_size, -half_range, half_range, grid_size, -half_range, half_range)

    safe_total_pt = max(total_pt, epsilon)

    # Fill histograms
    for eta_bin in range(grid_size):
        for phi_bin in range(grid_size):
            deta_center = -half_range + (eta_bin + 0.5) * bin_width
            dphi_center = -half_range + (phi_bin + 0.5) * bin_width
            pt_val = grid[eta_bin, phi_bin]

            fill_raw = pt_val if pt_val > 0 else floor_value
            norm_val = pt_val / safe_total_pt
            fill_norm = norm_val if norm_val > 0 else floor_value

            hist_raw.Fill(deta_center, dphi_center, fill_raw)
            hist_norm.Fill(deta_center, dphi_center, fill_norm)

    print(f"✅ Processed event {i}")
    break  # Only one event

# ROOT style
ROOT.gStyle.SetNumberContours(100)
ROOT.gStyle.SetPalette(ROOT.kViridis)

# Plotting
canvas = ROOT.TCanvas("canvas", "", 1200, 600)
canvas.Divide(2, 1)

canvas.cd(1)
ROOT.gPad.SetRightMargin(0.15)
hist_raw.Draw("COLZ TEXT")

canvas.cd(2)
ROOT.gPad.SetRightMargin(0.15)
hist_norm.Draw("COLZ TEXT")

canvas.SaveAs("jet_image_single_event.png")
print("✅ Saved: jet_image_single_event.png")
"""
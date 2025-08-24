#This program creates the input root file for training the multi channel multi weight CNN (Currently 3 channels) 
#The input is a text file with absolute paths to each of the input root files, 
#where each root file is seperated into signal/background by their file name. 
import ROOT
import numpy as np
import math
import os
from DataFormats.FWLite import Events, Handle

ROOT.FWLiteEnabler.enable()

# --- Image configuration ---
grid_size = 10
half_range = 0.5
bin_width = (2 * half_range) / grid_size
floor_value = 1e-5

# --- Output ROOT setup ---
output_file = ROOT.TFile("jet_images_ak15_3channel_weighted.root", "RECREATE")
tree = ROOT.TTree("JetImageTree", "AK15 3-channel jet images and labels")

img_pt = np.zeros(grid_size * grid_size, dtype=np.float32)
img_ch = np.zeros(grid_size * grid_size, dtype=np.float32)
img_neu = np.zeros(grid_size * grid_size, dtype=np.float32)
event_weight = np.zeros(1, dtype=np.float32)
event_type = ROOT.std.string()

tree.Branch("img_pt", img_pt, f"img_pt[{grid_size * grid_size}]/F")
tree.Branch("img_charged", img_ch, f"img_charged[{grid_size * grid_size}]/F")
tree.Branch("img_neutral", img_neu, f"img_neutral[{grid_size * grid_size}]/F")
tree.Branch("event_weight", event_weight, "event_weight/F")
tree.Branch("event_type", event_type)

# --- Fixed per-event weights from table ---
fixed_weights = {
    "HT-100To200": 1.46874032,
    "HT-1200To2500": 0.12214361,
    "HT-200To400": 0.44510441,
    "HT-2500ToInf": 0.00049437,
    "HT-400To600": 5.38596839,
    "HT-600To800": 0.94780151,
    "HT-800To1200": 0.27825656
}

# --- Helpers ---
def get_event_type(filename):
    fname = os.path.basename(filename).lower()
    if "8b" in fname:
        return "8b"
    elif "4b" in fname:
        return "4b"
    elif "wjet" in fname:
        return "wjets"
    else:
        return "unknown"

def get_fixed_weight(filename):
    for key, weight in fixed_weights.items():
        if key in filename:
            return weight
    return 1.0  # default for signal or unknown

# --- Input file list ---
with open("input_2.txt") as f:
    file_paths = [line.strip() for line in f if line.strip()]

# --- Setup handles ---
ak15_handle = Handle("std::vector<pat::Jet>")
ak15_label = ("selectedPatJetsAK15PFCHS", "", "NANO")

pf_handle = Handle("std::vector<pat::PackedCandidate>")
pf_label = ("packedPFCandidates", "", "PAT")

# --- Loop over files ---
for path in file_paths:
    print(f"📁 Processing: {path}")
    try:
        events = Events(f"file:{path}")
    except Exception as e:
        print(f"⚠️ Skipping {path} — {e}")
        continue

    evt_type_str = get_event_type(path)
    weight = get_fixed_weight(path)

    for event in events:
        event.getByLabel(ak15_label, ak15_handle)
        jets = ak15_handle.product()
        if jets.size() == 0:
            continue

        lead_jet = max(jets, key=lambda j: j.pt())
        if lead_jet.pt() < 300:
            continue

        jet_eta, jet_phi = lead_jet.eta(), lead_jet.phi()

        event.getByLabel(pf_label, pf_handle)
        pf_cands = pf_handle.product()

        pt_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        ch_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        neu_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        for cand in pf_cands:
            deta = cand.eta() - jet_eta
            dphi = math.atan2(math.sin(cand.phi() - jet_phi), math.cos(cand.phi() - jet_phi))

            if abs(deta) > half_range or abs(dphi) > half_range:
                continue

            eta_bin = int((deta + half_range) / bin_width)
            phi_bin = int((dphi + half_range) / bin_width)

            if 0 <= eta_bin < grid_size and 0 <= phi_bin < grid_size:
                pt = cand.pt()
                pt_grid[eta_bin, phi_bin] += pt
                if cand.charge() != 0:
                    ch_grid[eta_bin, phi_bin] += pt
                else:
                    neu_grid[eta_bin, phi_bin] += pt

        pt_grid = np.where(pt_grid > 0, pt_grid, floor_value)
        ch_grid = np.where(ch_grid > 0, ch_grid, floor_value)
        neu_grid = np.where(neu_grid > 0, neu_grid, floor_value)

        img_pt[:] = pt_grid.flatten()
        img_ch[:] = ch_grid.flatten()
        img_neu[:] = neu_grid.flatten()
        event_type.replace(0, ROOT.std.string.npos, evt_type_str)
        event_weight[0] = weight

        tree.Fill()

# --- Finalize ---
output_file.Write()
output_file.Close()
print("✅ Jet image ROOT file with fixed weights saved as jet_images_ak15_3channel_weighted.root")

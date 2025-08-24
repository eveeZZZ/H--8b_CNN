#This is to loop through a file, extract image and display what the average image for the file would look like. For AK15

"""
import ROOT
import math
import numpy as np
import matplotlib.pyplot as plt
from DataFormats.FWLite import Events, Handle

ROOT.FWLiteEnabler.enable()

# Input file
input_file = "/hdfs/store/user/abdollah/ExoHiggs_WithAK15/MiniAOD/8bjets/OUTROOT_8bjets_59_Mini.root"
events = Events(input_file)

# Handles
ak15_handle = Handle("std::vector<pat::Jet>")
ak15_label = ("selectedPatJetsAK15PFCHS", "", "NANO")

pf_handle = Handle("std::vector<pat::PackedCandidate>")
pf_label = ("packedPFCandidates", "", "PAT")

# Image grid settings
grid_size = 10
half_range = 0.5  # Δη, Δφ coverage
bin_width = (2 * half_range) / grid_size
image = np.zeros((grid_size, grid_size), dtype=np.float32)

# Helper function for Δφ wrapping
def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    while dphi > math.pi:
        dphi -= 2*math.pi
    while dphi < -math.pi:
        dphi += 2*math.pi
    return dphi

# Loop over events
for i, event in enumerate(events):
    event.getByLabel(ak15_label, ak15_handle)
    jets = ak15_handle.product()

    if len(jets) == 0:
        continue

    event.getByLabel(pf_label, pf_handle)
    pfs = pf_handle.product()

    # Leading jet
    leading_jet = sorted(jets, key=lambda j: j.pt(), reverse=True)[0]
    jet_eta = leading_jet.eta()
    jet_phi = leading_jet.phi()

    # Fill jet image
    for pf in pfs:
        deta = pf.eta() - jet_eta
        dphi = delta_phi(pf.phi(), jet_phi)

        if abs(deta) > half_range or abs(dphi) > half_range:
            continue

        ix = int((deta + half_range) / bin_width)
        iy = int((dphi + half_range) / bin_width)

        if 0 <= ix < grid_size and 0 <= iy < grid_size:
            image[ix, iy] += pf.pt()

    break  # Only process one event

# Plot image
plt.imshow(image.T, origin='lower', extent=[-half_range, half_range, -half_range, half_range],
           cmap='viridis', interpolation='nearest')
plt.colorbar(label='Sum pT')
plt.xlabel("Δη")
plt.ylabel("Δφ")
plt.title("Jet Image (AK15, leading jet)")
plt.grid(False)
plt.show()
"""
import ROOT
from DataFormats.FWLite import Events, Handle
import math

ROOT.FWLiteEnabler.enable()

# Input file
input_file = "/hdfs/store/user/abdollah/ExoHiggs_WithAK15/MiniAOD/8bjets/OUTROOT_8bjets_59_Mini.root"
events = Events(input_file)

# Handles and labels
ak15_handle = Handle("std::vector<pat::Jet>")
ak15_label = ("selectedPatJetsAK15PFCHS", "", "NANO")

pf_handle = Handle("std::vector<pat::PackedCandidate>")
pf_label = ("packedPFCandidates", "", "PAT")

# Parameters for jet image
grid_size = 10
half_range = 0.5
bin_width = (2 * half_range) / grid_size

# Process first event with jets
for i, event in enumerate(events):
    event.getByLabel(ak15_label, ak15_handle)
    jets = ak15_handle.product()

    if len(jets) == 0:
        continue

    event.getByLabel(pf_label, pf_handle)
    pfs = pf_handle.product()

    # Get leading AK15 jet
    leading_jet = max(jets, key=lambda j: j.pt())
    jet_eta = leading_jet.eta()
    jet_phi = leading_jet.phi()

    # Create TH2F image
    hist = ROOT.TH2F("jet_image", "AK15 Jet Image;#Delta#eta;#Delta#phi",
                     grid_size, -half_range, half_range,
                     grid_size, -half_range, half_range)

    # Helper for Δφ wrapping
    def delta_phi(phi1, phi2):
        dphi = phi1 - phi2
        while dphi > math.pi: dphi -= 2 * math.pi
        while dphi < -math.pi: dphi += 2 * math.pi
        return dphi

    # Fill image
    for pf in pfs:
        d_eta = pf.eta() - jet_eta
        d_phi = delta_phi(pf.phi(), jet_phi)

        if abs(d_eta) < half_range and abs(d_phi) < half_range:
            hist.Fill(d_eta, d_phi, pf.pt())

    # Draw and save
    canvas = ROOT.TCanvas("c", "", 800, 600)
    hist.SetStats(0)
    hist.Draw("COLZ")
    canvas.SaveAs("ak15_jet_image.png")
    print("✅ Saved jet image to ak15_jet_image.png")
    break  # Only process one event

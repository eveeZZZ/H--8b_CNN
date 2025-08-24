#The most successful Model as of now. 
#This model takes an input of 3 channels images root file, weight them, and train the CNN. 
#The program also produces an output file, a model, and a ROC Curve after predicting. 
import uproot
import numpy as np
import tensorflow as tf
import ROOT
from array import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# --- Load Data ---
file_path = "jet_images_ak15_3channel_weighted.root"
tree = uproot.open(file_path)["JetImageTree"]

img_pt = tree["img_pt"].array(library="np").reshape(-1, 10, 10)
img_ch = tree["img_charged"].array(library="np").reshape(-1, 10, 10)
img_neu = tree["img_neutral"].array(library="np").reshape(-1, 10, 10)
labels = tree["event_type"].array(library="np")
weights = tree["event_weight"].array(library="np")

# --- Filter only 8b and wjets ---
labels = np.array([l.decode() if isinstance(l, bytes) else l for l in labels])
mask = np.isin(labels, ["8b", "wjets"])
img_pt = img_pt[mask]
img_ch = img_ch[mask]
img_neu = img_neu[mask]
labels = labels[mask]
weights = weights[mask]

# --- Encode labels ---
label_map = {"wjets": 0, "8b": 1}
y = np.array([label_map[l] for l in labels])

# --- Stack as 3-channel input ---
X = np.stack([img_pt, img_ch, img_neu], axis=-1).astype(np.float32)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42, stratify=y
)

# --- Scale up signal (8b) weights ---
scale_factor = 20.0
scaled_w_train = np.where(y_train == 1, w_train * scale_factor, w_train)

# --- Model definition ---
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(10, 10, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# --- Compile with weighted AUC ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[],
    weighted_metrics=[tf.keras.metrics.AUC(name="weighted_auc")]
)

# --- Early stopping ---
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# --- Train ---
model.fit(
    X_train, y_train,
    sample_weight=scaled_w_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop]
)

# --- Save model ---
model.save("cnn_classifier_ak15_3channel_weighted_signalup.h5")

# --- Predict ---
y_score = model.predict(X_test).flatten()

# --- Save predictions to ROOT ---
output = ROOT.TFile("cnn_predictions_ak15_3channel_weighted_signalup.root", "RECREATE")
tree_out = ROOT.TTree("PredictionTree", "CNN predictions (3-channel, signal upweighted)")

label = array('i', [0])
score = array('f', [0.0])
tree_out.Branch("label", label, "label/I")
tree_out.Branch("score_8b", score, "score_8b/F")

for i in range(len(y_score)):
    label[0] = int(y_test[i])
    score[0] = float(y_score[i])
    tree_out.Fill()

output.Write()
output.Close()
print("✅ Predictions saved to cnn_predictions_ak15_3channel_weighted_signalup.root")

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
weighted_roc_auc = roc_auc_score(y_test, y_score, sample_weight=w_test)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f} (unweighted)\nAUC = {weighted_roc_auc:.3f} (weighted)")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: AK15 (8b vs. WJets, 3-channel, signal upweighted)")
plt.legend()
plt.grid(True)
plt.savefig("cnn_roc_ak15_3channel_weighted_signalup.png")
print("📈 ROC curve saved as cnn_roc_ak15_3channel_weighted_signalup.png")

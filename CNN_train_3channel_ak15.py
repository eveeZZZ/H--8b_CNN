#CNN with 3 channels, ak15 images as input
import uproot
import numpy as np
import tensorflow as tf
import ROOT
from array import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# --- Load Data ---
file_path = "jet_images_ak15_3channel_300.root"
tree = uproot.open(file_path)["JetImageTree"]

img_pt = tree["img_pt"].array(library="np").reshape(-1, 10, 10)
img_ch = tree["img_charged"].array(library="np").reshape(-1, 10, 10)
img_neu = tree["img_neutral"].array(library="np").reshape(-1, 10, 10)
labels = tree["event_type"].array(library="np")

# --- Filter only 8b and wjets ---
labels = np.array([l.decode() if isinstance(l, bytes) else l for l in labels])
mask = np.isin(labels, ["8b", "wjets"])
img_pt = img_pt[mask]
img_ch = img_ch[mask]
img_neu = img_neu[mask]
labels = labels[mask]

# --- Encode labels ---
label_map = {"wjets": 0, "8b": 1}
y = np.array([label_map[l] for l in labels])

# --- Stack as 3-channel input ---
X = np.stack([img_pt, img_ch, img_neu], axis=-1).astype(np.float32)  # (N, 10, 10, 3)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- CNN Model ---
model4 = tf.keras.Sequential([
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

model5 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(10, 10, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model= model4 #also 6
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy',metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])


from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
print("Original weights:", weights)

# Scale them halfway to 1
weights_scaled = 1 + 0.5 * (weights - 1)
class_weight_dict = dict(enumerate(weights_scaled))
print("Adjusted weights:", class_weight_dict)

#early stop
early_stop = EarlyStopping(
    monitor='val_loss',      # what to watch
    patience=3,              # stop if no improvement after 3 epochs
    restore_best_weights=True  # revert to best model
)

# --- Train ---
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,#64
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stop]
)
model.save("cnn_classifier_ak15_3channel_6_300_64.h5")

# --- Predict ---
y_score = model.predict(X_test).flatten()

# --- Save Predictions to ROOT ---
output = ROOT.TFile("cnn_predictions_ak15_3channel_6_300_64.root", "RECREATE")
tree_out = ROOT.TTree("PredictionTree", "CNN predictions (3-channel)")

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
print("✅ Predictions saved to cnn_predictions_ak15_3channel_6_300_64.root")

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: AK15 (8b vs. wjets, 3-channel)")
plt.legend()
plt.grid(True)
plt.savefig("cnn_roc_ak15_3channel_6_300_64.png")
print("📈 ROC curve saved as cnn_roc_ak15_3channel.png")

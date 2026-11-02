import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/agent2_processed/autoencoder_data.npz"
MODEL_PATH = "models/autoencoder.pth"
RESULT_DIR = "results"

BATCH_SIZE = 1024
DEVICE = torch.device("cpu")

os.makedirs(RESULT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading Agent 2 data...")

data = np.load(DATA_PATH)

X_val = data["X_val_all"].astype(np.float32)
y_val_original = data["y_val"]

X_test = data["X_test"].astype(np.float32)
y_test_original = data["y_test"]

normal_class = int(data["normal_class"])

# Binary labels: 0 = Normal, 1 = Attack
y_val = (y_val_original != normal_class).astype(int)
y_test = (y_test_original != normal_class).astype(int)

print("Validation samples:", X_val.shape)
print("Test samples:", X_test.shape)

# --------------------------------------------------
# 3. LOAD CLASS MAPPING
# --------------------------------------------------

with open("data/agent2_processed/class_mapping.json") as f:
    class_mapping = json.load(f)

# --------------------------------------------------
# 4. EXACT EXISTING AUTOENCODER ARCHITECTURE
# --------------------------------------------------

class Autoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


model = Autoencoder(
    input_dim=X_val.shape[1],
    hidden_dim=24,
    latent_dim=12
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("\nExisting Autoencoder checkpoint loaded successfully.")

# --------------------------------------------------
# 5. RECONSTRUCTION ERROR FUNCTION
# --------------------------------------------------

def get_reconstruction_errors(data_array):

    dataset = TensorDataset(
        torch.tensor(data_array, dtype=torch.float32)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    errors = []

    with torch.no_grad():

        for (features,) in loader:

            features = features.to(DEVICE)

            reconstructed = model(features)

            batch_errors = torch.mean(
                (features - reconstructed) ** 2,
                dim=1
            )

            errors.extend(batch_errors.cpu().numpy())

    return np.array(errors)

# --------------------------------------------------
# 6. VALIDATION THRESHOLD CALIBRATION
# --------------------------------------------------

print("\nCalculating validation reconstruction errors...")

val_errors = get_reconstruction_errors(X_val)

precision_values, recall_values, thresholds = (
    precision_recall_curve(y_val, val_errors)
)

# Calculate F1 for every available threshold.
denominator = precision_values[:-1] + recall_values[:-1]

f1_values = np.divide(
    2 * precision_values[:-1] * recall_values[:-1],
    denominator,
    out=np.zeros_like(denominator),
    where=denominator != 0
)

best_index = int(np.argmax(f1_values))

threshold = float(thresholds[best_index])

print("\n========== VALIDATION CALIBRATION ==========")
print("Selected threshold:", threshold)
print("Validation precision:", precision_values[best_index])
print("Validation recall:", recall_values[best_index])
print("Validation F1:", f1_values[best_index])

# --------------------------------------------------
# 7. TEST EVALUATION
# --------------------------------------------------

print("\nEvaluating on untouched test data...")

# Warm-up
warmup = torch.tensor(
    X_test[:min(256, len(X_test))],
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():
    _ = model(warmup)

start_time = time.perf_counter()

test_errors = get_reconstruction_errors(X_test)

inference_time = time.perf_counter() - start_time

predictions = (test_errors > threshold).astype(int)

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

cm = confusion_matrix(
    y_test, predictions, labels=[0, 1]
)

# --------------------------------------------------
# 8. PER-CLASS DETECTION RATES
# --------------------------------------------------

class_detection_rates = {}

for class_id, class_name in class_mapping.items():

    class_id = int(class_id)
    mask = y_test_original == class_id

    if np.sum(mask) == 0:
        continue

    rate = float(np.mean(predictions[mask]) * 100)

    class_detection_rates[class_name] = round(rate, 2)

# --------------------------------------------------
# 9. PERFORMANCE MEASUREMENTS
# --------------------------------------------------

total_samples = len(X_test)

average_latency_ms = (
    inference_time / total_samples
) * 1000

throughput = total_samples / inference_time

model_size_mb = (
    os.path.getsize(MODEL_PATH) / (1024 * 1024)
)

# Preserve the previously recorded training time.
training_time = None

old_performance_path = os.path.join(
    RESULT_DIR, "agent2_performance.json"
)

if os.path.exists(old_performance_path):

    with open(old_performance_path) as f:
        old_performance = json.load(f)

    training_time = old_performance.get(
        "training_time_seconds"
    )

performance = {
    "agent": "Autoencoder",
    "training_time_seconds": training_time,
    "inference_time_seconds": round(inference_time, 4),
    "average_latency_ms": round(average_latency_ms, 6),
    "throughput_samples_per_second": round(throughput, 2),
    "model_size_mb": round(model_size_mb, 4),
    "threshold": threshold,
    "threshold_selection": "Maximum validation F1-score",
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "class_anomaly_detection_rates": class_detection_rates
}

# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

np.savez_compressed(
    os.path.join(RESULT_DIR, "autoencoder_predictions.npz"),
    y_true=y_test,
    y_pred=predictions,
    reconstruction_errors=test_errors
)

with open(
    os.path.join(RESULT_DIR, "autoencoder_threshold.json"), "w"
) as f:

    json.dump(
        {
            "threshold": threshold,
            "selection_method": "Maximum validation F1-score",
            "validation_f1": float(f1_values[best_index])
        },
        f,
        indent=4
    )

with open(
    old_performance_path, "w"
) as f:

    json.dump(performance, f, indent=4)

# Save confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"]
)

plt.title("Agent 2 - Autoencoder Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "autoencoder_confusion_matrix.png"
    )
)

plt.close()

# --------------------------------------------------
# 11. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\n" + "=" * 50)
print("AGENT 2 - CALIBRATED TEST RESULTS")
print("=" * 50)

print(f"Threshold: {threshold:.8f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nPer-class anomaly detection rates:")

for name, rate in class_detection_rates.items():
    print(f"{name}: {rate:.2f}%")

print("\nUpdated results saved in:", RESULT_DIR)
print("No model retraining performed.")
print("\nAGENT 2 CALIBRATION COMPLETED!")
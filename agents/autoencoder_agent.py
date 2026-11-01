
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/agent2_processed/autoencoder_data.npz"
MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

BATCH_SIZE = 1024
EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 3

INPUT_DIM = 71
HIDDEN_DIM = 24
LATENT_DIM = 12
#requirements slower below is slow
#BATCH_SIZE = 512
#EPOCHS = 30
#LEARNING_RATE = 0.001
#PATIENCE = 5

#INPUT_DIM = 71
#HIDDEN_DIM = 32
#LATENT_DIM = 16

# Threshold based on the 99th percentile of
# reconstruction errors on normal validation traffic.
THRESHOLD_PERCENTILE = 99

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------
# 2. LOAD PREPROCESSED DATA
# --------------------------------------------------

print("\nLoading preprocessed dataset...")

data = np.load(DATA_PATH)

X_train_normal = data["X_train_normal"].astype(np.float32)
X_val_normal = data["X_val_normal"].astype(np.float32)

X_test = data["X_test"].astype(np.float32)
y_test = data["y_test"]

num_features = X_train_normal.shape[1]

print("Normal training samples:", X_train_normal.shape)
print("Normal validation samples:", X_val_normal.shape)
print("Testing samples:", X_test.shape)
print("Features:", num_features)

# Identify Normal class using the saved class mapping.
with open("data/processed/class_mapping.json") as f:
    class_mapping = json.load(f)

normal_class = int(
    [key for key, value in class_mapping.items()
     if value.strip().lower() == "normal"][0]
)

print("Normal class index:", normal_class)


# --------------------------------------------------
# 3. DATA LOADERS
# --------------------------------------------------

train_dataset = TensorDataset(
    torch.tensor(X_train_normal, dtype=torch.float32)
)

val_dataset = TensorDataset(
    torch.tensor(X_val_normal, dtype=torch.float32)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# 4. AUTOENCODER MODEL
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
    input_dim=num_features,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM
).to(DEVICE)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# 5. VALIDATION FUNCTION
# --------------------------------------------------

def calculate_validation_loss():

    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for (features,) in val_loader:

            features = features.to(DEVICE)

            reconstructed = model(features)

            loss = criterion(reconstructed, features)

            total_loss += loss.item() * len(features)

    return total_loss / len(val_loader.dataset)


# --------------------------------------------------
# 6. TRAIN AUTOENCODER
# --------------------------------------------------

best_val_loss = float("inf")
epochs_without_improvement = 0

history = {
    "train_loss": [],
    "val_loss": []
}

print("\nStarting Autoencoder training...")

training_start_time = time.perf_counter()

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0.0

    for (features,) in train_loader:

        features = features.to(DEVICE)

        optimizer.zero_grad()

        reconstructed = model(features)

        loss = criterion(reconstructed, features)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(features)

    train_loss = total_loss / len(train_loader.dataset)

    val_loss = calculate_validation_loss()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Validation Loss: {val_loss:.6f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, "autoencoder.pth")
        )

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("Early stopping triggered.")
        break


training_end_time = time.perf_counter()

training_time = training_end_time - training_start_time

print(f"\nTraining time: {training_time:.2f} seconds")
print(f"Training time: {training_time / 60:.2f} minutes")


# Save training history
with open(
    os.path.join(RESULT_DIR, "autoencoder_history.json"), "w"
) as f:
    json.dump(history, f, indent=4)


# --------------------------------------------------
# 7. LOAD BEST MODEL
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "autoencoder.pth"),
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()


# --------------------------------------------------
# 8. RECONSTRUCTION ERROR FUNCTION
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
# 9. CALCULATE ANOMALY THRESHOLD
# --------------------------------------------------

print("\nCalculating threshold using normal validation data...")

val_errors = get_reconstruction_errors(X_val_normal)

threshold = float(
    np.percentile(val_errors, THRESHOLD_PERCENTILE)
)

print(f"Threshold percentile: {THRESHOLD_PERCENTILE}")
print(f"Anomaly threshold: {threshold:.8f}")


# --------------------------------------------------
# 10. TEST EVALUATION AND SPEED
# --------------------------------------------------

# Warm-up
warmup = torch.tensor(
    X_test[:min(256, len(X_test))],
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():
    _ = model(warmup)

print("\nEvaluating Autoencoder on test data...")

inference_start_time = time.perf_counter()

test_errors = get_reconstruction_errors(X_test)

inference_end_time = time.perf_counter()

inference_time = inference_end_time - inference_start_time

total_samples = len(X_test)

average_latency_ms = (
    inference_time / total_samples
) * 1000

throughput = total_samples / inference_time

# Higher reconstruction error means anomaly.
anomaly_predictions = (test_errors > threshold).astype(int)

# Ground truth: 0 = Normal, 1 = Attack.
y_binary = (y_test != normal_class).astype(int)

accuracy = accuracy_score(y_binary, anomaly_predictions)

precision = precision_score(
    y_binary,
    anomaly_predictions,
    zero_division=0
)

recall = recall_score(
    y_binary,
    anomaly_predictions,
    zero_division=0
)

f1 = f1_score(
    y_binary,
    anomaly_predictions,
    zero_division=0
)


# --------------------------------------------------
# 11. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("AGENT 2 - AUTOENCODER RESULTS")
print("=" * 55)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nBinary Classification Report:\n")

print(
    classification_report(
        y_binary,
        anomaly_predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_binary, anomaly_predictions))


# --------------------------------------------------
# 12. PER-ATTACK ANOMALY DETECTION RATES
# --------------------------------------------------

print("\nPer-class anomaly detection rates:")

class_detection_rates = {}

for class_id, class_name in class_mapping.items():

    class_id = int(class_id)

    mask = y_test == class_id

    if np.sum(mask) == 0:
        continue

    rate = float(
        np.mean(anomaly_predictions[mask]) * 100
    )

    class_detection_rates[class_name] = round(rate, 2)

    print(f"{class_name}: {rate:.2f}%")


# --------------------------------------------------
# 13. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_binary,
    anomaly_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Anomaly"],
    yticklabels=["Normal", "Anomaly"]
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
# 14. SAVE PREDICTIONS AND RECONSTRUCTION ERRORS
# --------------------------------------------------

np.savez_compressed(
    os.path.join(RESULT_DIR, "autoencoder_predictions.npz"),
    y_true=y_binary,
    y_pred=anomaly_predictions,
    reconstruction_errors=test_errors
)


# Save threshold separately
with open(
    os.path.join(RESULT_DIR, "autoencoder_threshold.json"), "w"
) as f:

    json.dump(
        {
            "threshold": threshold,
            "percentile": THRESHOLD_PERCENTILE
        },
        f,
        indent=4
    )


# --------------------------------------------------
# 15. PERFORMANCE MEASUREMENTS
# --------------------------------------------------

model_path = os.path.join(MODEL_DIR, "autoencoder.pth")

model_size_mb = (
    os.path.getsize(model_path) / (1024 * 1024)
)

performance = {
    "agent": "Autoencoder",
    "training_time_seconds": round(training_time, 4),
    "inference_time_seconds": round(inference_time, 4),
    "average_latency_ms": round(average_latency_ms, 6),
    "throughput_samples_per_second": round(throughput, 2),
    "model_size_mb": round(model_size_mb, 4),
    "threshold": threshold,
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "class_anomaly_detection_rates": class_detection_rates
}

with open(
    os.path.join(RESULT_DIR, "agent2_performance.json"), "w"
) as f:

    json.dump(performance, f, indent=4)


# --------------------------------------------------
# 16. FINAL OUTPUT
# --------------------------------------------------

print("\n========== AGENT 2 PERFORMANCE ==========")
print(f"Training time: {training_time:.2f} seconds")
print(f"Inference time: {inference_time:.4f} seconds")
print(f"Average latency: {average_latency_ms:.6f} ms/sample")
print(f"Throughput: {throughput:.2f} samples/second")
print(f"Model size: {model_size_mb:.2f} MB")

print("\nModel saved: models/autoencoder.pth")
print("Results saved in:", RESULT_DIR)
print("\nAGENT 2 TRAINING COMPLETED!")

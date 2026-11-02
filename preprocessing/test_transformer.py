
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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/processed/dataset.npz"
MODEL_PATH = "models/ft_transformer.pth"
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")
BATCH_SIZE = 512

torch.manual_seed(42)
np.random.seed(42)

# --------------------------------------------------
# 1. LOAD TEST DATA
# --------------------------------------------------

print("\nLoading processed dataset...")

data = np.load(DATA_PATH)

X_test = data["X_test"]
y_test = data["y_test"]

num_features = X_test.shape[1]
num_classes = len(np.unique(data["y_train"]))

print("Test samples:", X_test.shape)
print("Features:", num_features)
print("Classes:", num_classes)

# --------------------------------------------------
# 2. DEFINE SAME MODEL ARCHITECTURE
# --------------------------------------------------

class FTTransformer(nn.Module):

    def __init__(
        self,
        num_features,
        num_classes,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.feature_weight = nn.Parameter(
            torch.randn(num_features, d_model) * 0.02
        )

        self.feature_bias = nn.Parameter(
            torch.zeros(num_features, d_model)
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = x.unsqueeze(-1) * self.feature_weight
        x = x + self.feature_bias

        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )

        x = torch.cat([cls_tokens, x], dim=1)

        x = self.transformer(x)

        x = self.norm(x[:, 0])

        return self.classifier(x)


# --------------------------------------------------
# 3. LOAD EXISTING TRAINED MODEL
# --------------------------------------------------

model = FTTransformer(
    num_features=num_features,
    num_classes=num_classes,
    d_model=64,
    num_heads=4,
    num_layers=2,
    dropout=0.2
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("\nExisting trained model loaded successfully.")

# --------------------------------------------------
# 4. TEST AND MEASURE INFERENCE TIME
# --------------------------------------------------

X_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

all_predictions = []
all_probabilities = []

# Warm-up
with torch.no_grad():
    _ = model(X_tensor[:min(256, len(X_tensor))])

inference_start = time.perf_counter()

with torch.no_grad():

    for start in range(0, len(X_test), BATCH_SIZE):

        batch = X_tensor[start:start + BATCH_SIZE]

        outputs = model(batch)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(
            probabilities, dim=1
        )

        all_predictions.extend(
            predictions.numpy()
        )

        all_probabilities.extend(
            probabilities.numpy()
        )

inference_end = time.perf_counter()

all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)

inference_time = inference_end - inference_start

total_samples = len(X_test)

average_latency_ms = (
    inference_time / total_samples
) * 1000

throughput = total_samples / inference_time

# --------------------------------------------------
# 5. PERFORMANCE METRICS
# --------------------------------------------------

accuracy = accuracy_score(
    y_test, all_predictions
)

precision = precision_score(
    y_test,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    all_predictions,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    all_predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    all_predictions,
    average="weighted",
    zero_division=0
)

model_size_mb = os.path.getsize(
    MODEL_PATH
) / (1024 * 1024)

# --------------------------------------------------
# 6. CLASSIFICATION REPORT
# --------------------------------------------------

with open(
    "data/processed/class_mapping.json",
    "r"
) as f:

    class_mapping = json.load(f)

target_names = [
    class_mapping[str(i)]
    for i in range(num_classes)
]

report = classification_report(
    y_test,
    all_predictions,
    labels=list(range(num_classes)),
    target_names=target_names,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

# --------------------------------------------------
# 7. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    all_predictions,
    labels=list(range(num_classes))
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.title("Agent 1 - FT-Transformer Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "transformer_confusion_matrix.png"
    )
)

plt.close()

# --------------------------------------------------
# 8. SAVE PREDICTIONS
# --------------------------------------------------

np.savez_compressed(
    os.path.join(
        RESULT_DIR,
        "transformer_predictions.npz"
    ),
    y_true=y_test,
    y_pred=all_predictions,
    probabilities=all_probabilities
)

# --------------------------------------------------
# 9. SAVE PERFORMANCE
# --------------------------------------------------

performance = {
    "agent": "FT-Transformer",
    "inference_time_seconds": round(inference_time, 4),
    "average_latency_ms": round(average_latency_ms, 6),
    "throughput_samples_per_second": round(throughput, 2),
    "model_size_mb": round(model_size_mb, 4),
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "macro_f1": round(macro_f1, 4),
    "weighted_f1": round(weighted_f1, 4)
}

with open(
    os.path.join(
        RESULT_DIR,
        "agent1_performance.json"
    ),
    "w"
) as f:

    json.dump(performance, f, indent=4)

# --------------------------------------------------
# 10. DISPLAY RESULTS
# --------------------------------------------------

print("\n========== AGENT 1 PERFORMANCE ==========")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print(f"\nInference time: {inference_time:.4f} seconds")
print(f"Average latency: {average_latency_ms:.6f} ms/sample")
print(f"Throughput: {throughput:.2f} samples/second")
print(f"Model size: {model_size_mb:.4f} MB")

print("\nPredictions saved.")
print("Confusion matrix saved.")
print("Performance saved: results/agent1_performance.json")

print("\nAGENT 1 TESTING COMPLETED!")

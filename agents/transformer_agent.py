
import os
import json
import numpy as np
import torch
import torch.nn as nn
import time

from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = "data/processed/dataset.npz"
MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

#BATCH_SIZE = 256
#EPOCHS = 20
#LEARNING_RATE = 0.0005

#D_MODEL = 64
#NUM_HEADS = 4
#NUM_LAYERS = 2
#DROPOUT = 0.2

# Lightweight FT-Transformer configuration
D_MODEL = 32
NUM_HEADS = 2
NUM_LAYERS = 1
FF_DIM = 64
DROPOUT = 0.1

BATCH_SIZE = 512
EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 0.0005

torch.manual_seed(42)
np.random.seed(42)

# --------------------------------------------------
# 1. LOAD PREPROCESSED DATA
# --------------------------------------------------

print("\nLoading preprocessed dataset...")

data = np.load(DATA_PATH)

X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]

num_features = X_train.shape[1]
num_classes = len(np.unique(y_train))

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
print("Features:", num_features)
print("Classes:", num_classes)

# --------------------------------------------------
# 2. FT-TRANSFORMER MODEL
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

        self.num_features = num_features

        # Each numerical feature gets its own
        # learnable projection into embedding space.
        self.feature_weight = nn.Parameter(
            torch.randn(num_features, d_model) * 0.02
        )

        self.feature_bias = nn.Parameter(
            torch.zeros(num_features, d_model)
        )

        # Learnable classification token
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

        # Numerical feature tokenization
        x = x.unsqueeze(-1) * self.feature_weight
        x = x + self.feature_bias

        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )

        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer feature interaction
        x = self.transformer(x)

        # Use classification token
        x = self.norm(x[:, 0])

        return self.classifier(x)


# --------------------------------------------------
# 3. CREATE DATA LOADERS
# --------------------------------------------------

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)

test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --------------------------------------------------
# 4. INITIALIZE MODEL
# --------------------------------------------------

model = FTTransformer(
    num_features=num_features,
    num_classes=num_classes,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

# Balanced class weights
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# --------------------------------------------------
# 5. TRAINING FUNCTION
# --------------------------------------------------

def evaluate_loss(model, loader):

    model.eval()
    total_loss = 0

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


# --------------------------------------------------
# 6. TRAIN FT-TRANSFORMER
# --------------------------------------------------
training_start_time = time.perf_counter()
best_val_loss = float("inf")
patience = 5
epochs_without_improvement = 0

history = {
    "train_loss": [],
    "val_loss": []
}

print("\nStarting FT-Transformer training...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for features, labels in train_loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * len(labels)

    train_loss = total_loss / len(train_loader.dataset)

    val_loss = evaluate_loss(model, val_loader)

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            os.path.join(
                MODEL_DIR, "ft_transformer.pth"
            )
        )

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:

        print("Early stopping triggered.")
        break

# Save training history
training_end_time = time.perf_counter()

training_time = training_end_time - training_start_time

print(f"\nTraining time: {training_time:.2f} seconds")
print(f"Training time: {training_time / 60:.2f} minutes")
with open(
    os.path.join(RESULT_DIR, "transformer_history.json"),
    "w"
) as f:
    json.dump(history, f, indent=4)

# --------------------------------------------------
# 7. LOAD BEST MODEL
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "ft_transformer.pth"),
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()



# --------------------------------------------------
# 8. TEST EVALUATION
# --------------------------------------------------

# Warm-up before benchmarking
with torch.no_grad():

    warmup_input = torch.tensor(
        X_test[:min(256, len(X_test))],
        dtype=torch.float32
    ).to(DEVICE)

    _ = model(warmup_input)


# Start inference benchmark
inference_start_time = time.perf_counter()


all_predictions = []
all_probabilities = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(DEVICE)

        outputs = model(features)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(
            probabilities, dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)


# End inference benchmark
inference_end_time = time.perf_counter()

inference_time = inference_end_time - inference_start_time

total_samples = len(X_test)

average_latency_ms = (
    inference_time / total_samples
) * 1000

throughput = total_samples / inference_time

accuracy = accuracy_score(
    y_test, all_predictions
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

model_path = os.path.join(
    MODEL_DIR, "ft_transformer.pth"
)

model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

print("\n========== AGENT 1 PERFORMANCE ==========")
print(f"Training time: {training_time:.2f} seconds")
print(f"Inference time: {inference_time:.4f} seconds")
print(f"Average latency: {average_latency_ms:.4f} ms/sample")
print(f"Throughput: {throughput:.2f} samples/second")
print(f"Model size: {model_size_mb:.2f} MB")

print("\n" + "=" * 55)
print("AGENT 1 - FT-TRANSFORMER RESULTS")
print("=" * 55)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

# Load actual class names
with open(
    os.path.join(
        "data/processed", "class_mapping.json"
    )
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

print("\nClassification Report:\n")
print(report)

# --------------------------------------------------
# 9. CONFUSION MATRIX
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
        RESULT_DIR, "transformer_confusion_matrix.png"
    )
)

plt.close()

# --------------------------------------------------
# 10. SAVE PREDICTIONS
# --------------------------------------------------

np.savez_compressed(
    os.path.join(
        RESULT_DIR, "transformer_predictions.npz"
    ),
    y_true=y_test,
    y_pred=all_predictions,
    probabilities=all_probabilities
)

print("\nModel saved: models/ft_transformer.pth")
print("Results saved in:", RESULT_DIR)
print("\nAGENT 1 TRAINING COMPLETED!")
performance = {
    "agent": "FT-Transformer",
    "training_time_seconds": round(training_time, 4),
    "inference_time_seconds": round(inference_time, 4),
    "average_latency_ms": round(average_latency_ms, 6),
    "throughput_samples_per_second": round(throughput, 2),
    "model_size_mb": round(model_size_mb, 4),
    "accuracy": round(accuracy, 4),
    "macro_f1": round(macro_f1, 4),
    "weighted_f1": round(weighted_f1, 4)
}

with open(
    os.path.join(RESULT_DIR, "agent1_performance.json"), "w"
) as f:
    json.dump(performance, f, indent=4)

print("Performance metrics saved: results/agent1_performance.json")
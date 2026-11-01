
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = "data/raw/IDS2025.xlsx"
OUTPUT_DIR = "data/agent2_processed"

LABEL_COLUMN = "newLabel"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

print("\nLoading IDS2025 dataset for Agent 2...")

df = pd.read_excel(INPUT_FILE)

df.columns = df.columns.astype(str).str.strip()

df = df.dropna(subset=[LABEL_COLUMN]).copy()

df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

df = df.drop_duplicates().reset_index(drop=True)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. SEPARATE FEATURES AND LABELS
# --------------------------------------------------

y = df[LABEL_COLUMN]

X = df.drop(columns=[LABEL_COLUMN]).copy()

X = X.apply(pd.to_numeric, errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

all_missing = X.columns[X.isna().all()].tolist()

X = X.drop(columns=all_missing)

constant_cols = X.columns[
    X.nunique(dropna=True) <= 1
].tolist()

X = X.drop(columns=constant_cols)

feature_names = X.columns.tolist()

print("Features remaining:", len(feature_names))

# --------------------------------------------------
# 3. ENCODE LABELS
# --------------------------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

normal_index = int(
    np.where(label_encoder.classes_ == "Normal")[0][0]
)

print("\nClass mapping:")

for i, label in enumerate(label_encoder.classes_):
    print(i, "=", label)

# --------------------------------------------------
# 4. SAME TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("\nSplit sizes:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# --------------------------------------------------
# 5. IMPUTE MISSING VALUES
# --------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

# --------------------------------------------------
# 6. ROBUST SCALING
# --------------------------------------------------

robust_scaler = RobustScaler()

X_train = robust_scaler.fit_transform(X_train)
X_val = robust_scaler.transform(X_val)
X_test = robust_scaler.transform(X_test)

# --------------------------------------------------
# 7. SELECT NORMAL TRAINING SAMPLES
# --------------------------------------------------

X_train_normal = X_train[y_train == normal_index]

X_val_normal = X_val[y_val == normal_index]

print("\nNormal training samples:", len(X_train_normal))
print("Normal validation samples:", len(X_val_normal))

# --------------------------------------------------
# 8. QUANTILE TRANSFORMATION
# --------------------------------------------------

print("\nApplying Quantile Transformation...")

# Fit the transformation using only normal training traffic.
quantile_scaler = QuantileTransformer(
    n_quantiles=min(1000, len(X_train_normal)),
    output_distribution="normal",
    random_state=RANDOM_STATE,
    subsample=100000
)

quantile_scaler.fit(X_train_normal)

# Apply identical transformation to all splits.
X_train_normal_scaled = quantile_scaler.transform(
    X_train_normal
)

X_val_normal_scaled = quantile_scaler.transform(
    X_val_normal
)

X_test_scaled = quantile_scaler.transform(X_test)

# Clip extreme transformed values for numerical stability.
X_train_normal_scaled = np.clip(
    X_train_normal_scaled, -5, 5
)

X_val_normal_scaled = np.clip(
    X_val_normal_scaled, -5, 5
)

X_test_scaled = np.clip(
    X_test_scaled, -5, 5
)

# Convert to float32.
X_train_normal_scaled = X_train_normal_scaled.astype(np.float32)

X_val_normal_scaled = X_val_normal_scaled.astype(np.float32)

X_test_scaled = X_test_scaled.astype(np.float32)

# --------------------------------------------------
# 9. SAVE AGENT 2 DATA
# --------------------------------------------------

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "autoencoder_data.npz"),

    X_train_normal=X_train_normal_scaled,
    X_val_normal=X_val_normal_scaled,

    X_test=X_test_scaled,

    y_test=y_test,

    normal_class=normal_index
)

joblib.dump(
    imputer,
    os.path.join(OUTPUT_DIR, "autoencoder_imputer.pkl")
)

joblib.dump(
    robust_scaler,
    os.path.join(OUTPUT_DIR, "autoencoder_robust_scaler.pkl")
)

joblib.dump(
    quantile_scaler,
    os.path.join(OUTPUT_DIR, "autoencoder_quantile_scaler.pkl")
)

with open(
    os.path.join(OUTPUT_DIR, "class_mapping.json"), "w"
) as f:

    json.dump(
        {
            int(i): str(label)
            for i, label in enumerate(label_encoder.classes_)
        },
        f,
        indent=4
    )

with open(
    os.path.join(OUTPUT_DIR, "feature_names.json"), "w"
) as f:

    json.dump(feature_names, f, indent=4)

# --------------------------------------------------
# 10. FINAL CHECK
# --------------------------------------------------

print("\n" + "=" * 55)
print("AGENT 2 PREPROCESSING COMPLETED")
print("=" * 55)

print("Normal training:", X_train_normal_scaled.shape)
print("Normal validation:", X_val_normal_scaled.shape)
print("Testing:", X_test_scaled.shape)

print("Minimum:", X_train_normal_scaled.min())
print("Maximum:", X_train_normal_scaled.max())

print("\nSaved in:", OUTPUT_DIR)
print("=" * 55)

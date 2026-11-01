
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer

# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------

INPUT_FILE = "data/raw/IDS2025.xlsx"
OUTPUT_DIR = "data/processed"
LABEL_COLUMN = "newLabel"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------

print("\nLoading IDS2025 dataset...")

df = pd.read_excel(INPUT_FILE)
df.columns = df.columns.astype(str).str.strip()

print("Original shape:", df.shape)

if LABEL_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{LABEL_COLUMN}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# ------------------------------------------------
# 2. CLEAN LABELS AND DUPLICATES
# ------------------------------------------------

df = df.dropna(subset=[LABEL_COLUMN]).copy()

df[LABEL_COLUMN] = (
    df[LABEL_COLUMN].astype(str).str.strip()
)

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)

print("Duplicates removed:", before - len(df))
print("Shape after duplicate removal:", df.shape)

# ------------------------------------------------
# 3. SEPARATE FEATURES AND LABELS
# ------------------------------------------------

y = df[LABEL_COLUMN]

X = df.drop(columns=[LABEL_COLUMN]).copy()

# Ensure all feature columns are numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values with missing values
X = X.replace([np.inf, -np.inf], np.nan)

# Remove columns that contain no usable values
all_missing = X.columns[X.isna().all()].tolist()
X = X.drop(columns=all_missing)

# Remove constant features
constant_cols = X.columns[
    X.nunique(dropna=True) <= 1
].tolist()

X = X.drop(columns=constant_cols)

print("All-missing columns removed:", all_missing)
print("Constant columns removed:", constant_cols)
print("Remaining features:", X.shape[1])

# ------------------------------------------------
# 4. ENCODE LABELS
# ------------------------------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClass mapping:")

for index, label in enumerate(label_encoder.classes_):
    print(index, "=", label)

# ------------------------------------------------
# 5. TRAIN / VALIDATION / TEST SPLIT
# ------------------------------------------------

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

# ------------------------------------------------
# 6. IMPUTE MISSING VALUES
# ------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

# ------------------------------------------------
# 7. SCALE FEATURES
# ------------------------------------------------

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Use float32 for efficient neural network training
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)

# ------------------------------------------------
# 8. PREPARE NORMAL-ONLY DATA FOR AGENT 2
# ------------------------------------------------

normal_label = "Normal"

if normal_label not in label_encoder.classes_:
    raise ValueError("Normal class not found in dataset.")

normal_index = int(
    np.where(label_encoder.classes_ == normal_label)[0][0]
)

X_train_normal = X_train[y_train == normal_index]
X_val_normal = X_val[y_val == normal_index]

if len(X_train_normal) == 0 or len(X_val_normal) == 0:
    raise ValueError("Insufficient normal samples for Autoencoder.")

# ------------------------------------------------
# 9. SAVE ALL DATA
# ------------------------------------------------

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "dataset.npz"),
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    X_train_normal=X_train_normal,
    X_val_normal=X_val_normal
)

joblib.dump(
    imputer,
    os.path.join(OUTPUT_DIR, "imputer.pkl")
)

joblib.dump(
    scaler,
    os.path.join(OUTPUT_DIR, "scaler.pkl")
)

joblib.dump(
    label_encoder,
    os.path.join(OUTPUT_DIR, "label_encoder.pkl")
)

feature_names = X.columns.tolist()

with open(
    os.path.join(OUTPUT_DIR, "feature_names.json"),
    "w"
) as f:
    json.dump(feature_names, f, indent=4)

with open(
    os.path.join(OUTPUT_DIR, "class_mapping.json"),
    "w"
) as f:
    json.dump(
        {int(i): str(label)
         for i, label in enumerate(label_encoder.classes_)},
        f,
        indent=4
    )

# ------------------------------------------------
# 10. FINAL SUMMARY
# ------------------------------------------------

print("\n" + "=" * 55)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 55)

print("Final features:", len(feature_names))
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nAgent 2 normal-only data:")
print("Normal training samples:", len(X_train_normal))
print("Normal validation samples:", len(X_val_normal))

print("\nSaved files in:", OUTPUT_DIR)
print("=" * 55)

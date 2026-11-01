
import numpy as np

data = np.load("data/processed/dataset.npz")

X_train = data["X_train"]
X_train_normal = data["X_train_normal"]
X_val_normal = data["X_val_normal"]
X_test = data["X_test"]

print("\n========== DATA RANGE CHECK ==========")

for name, array in [
    ("Training", X_train),
    ("Normal Training", X_train_normal),
    ("Normal Validation", X_val_normal),
    ("Testing", X_test)
]:
    print(f"\n{name}")
    print("Minimum:", np.min(array))
    print("Maximum:", np.max(array))
    print("Mean:", np.mean(array))
    print("Standard deviation:", np.std(array))
    print("NaN values:", np.isnan(array).sum())
    print("Infinite values:", np.isinf(array).sum())

    print(
        "99th percentile of absolute values:",
        np.percentile(np.abs(array), 99)
    )

print("\nData inspection completed.")

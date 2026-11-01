
import numpy as np

data = np.load(
    "data/agent2_processed/autoencoder_data.npz"
)

for name in [
    "X_train_normal",
    "X_val_normal",
    "X_test"
]:
    array = data[name]

    print(f"\n{name}")
    print("Shape:", array.shape)
    print("Minimum:", array.min())
    print("Maximum:", array.max())
    print("Mean:", array.mean())
    print("Standard deviation:", array.std())
    print("NaN:", np.isnan(array).sum())
    print("Infinite:", np.isinf(array).sum())


import json
import numpy as np

# Load saved reconstruction errors
predictions = np.load(
    "results/autoencoder_predictions.npz"
)

errors = predictions["reconstruction_errors"]

# Load actual test labels
data = np.load(
    "data/agent2_processed/autoencoder_data.npz"
)

y_test = data["y_test"]

# Load class names
with open(
    "data/agent2_processed/class_mapping.json"
) as f:
    class_mapping = json.load(f)

# Load threshold
with open(
    "results/autoencoder_threshold.json"
) as f:
    threshold = json.load(f)["threshold"]

print("\n========== RECONSTRUCTION ERROR ANALYSIS ==========")

print("Threshold:", threshold)

normal_class = int(data["normal_class"])

for class_id, class_name in class_mapping.items():

    class_id = int(class_id)

    class_errors = errors[y_test == class_id]

    if len(class_errors) == 0:
        continue

    print(f"\n{class_name}")
    print("Samples:", len(class_errors))
    print("Mean error:", np.mean(class_errors))
    print("Median error:", np.median(class_errors))
    print("90th percentile:", np.percentile(class_errors, 90))
    print("95th percentile:", np.percentile(class_errors, 95))
    print("99th percentile:", np.percentile(class_errors, 99))
    print(
        "Detected as anomaly (%):",
        np.mean(class_errors > threshold) * 100
    )

print("\nAnalysis completed.")

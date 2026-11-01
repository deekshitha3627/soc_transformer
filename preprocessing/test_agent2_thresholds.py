
import json
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

# Load errors and labels
pred = np.load("results/autoencoder_predictions.npz")

errors = pred["reconstruction_errors"]
y_true = pred["y_true"]

# Test different thresholds using existing test errors
# Diagnostic only: these results must NOT be used
# to select the final test threshold.

print("\nTHRESHOLD SENSITIVITY ANALYSIS")
print("Diagnostic only - not for final model selection\n")

for percentile in [90, 95, 97, 98, 99]:

    threshold = np.percentile(errors[y_true == 0], percentile)

    predictions = (errors > threshold).astype(int)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    false_positive_rate = np.mean(
        predictions[y_true == 0]
    )

    print(f"Percentile: {percentile}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"False positive rate: {false_positive_rate:.4f}")
    print("-" * 35)

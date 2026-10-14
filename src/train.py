"""
Step 4: Train the lightweight CNN on FFT features and, separately, on
wavelet features. Produces two FP32 baseline models.

Run: python src/train.py            (clean training features)
     python src/train.py --aug      (noise-augmented features; run
                                     src/augment_train.py first)

Output (clean): results/{fft,wavelet}_model_fp32.pt, results/train_log.csv
Output (--aug): results/{fft,wavelet}_model_fp32_aug.pt, results/train_log_aug.csv

Only the TRAIN split is swapped for its augmented version; val and test stay
clean so the two runs are compared on an identical yardstick.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from model import LightweightCNN1D, count_parameters

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

EPOCHS = 100
PATIENCE = 15  # early stopping
BATCH_SIZE = 32
LR = 1e-3


def load_split(feature_type, split, suffix=""):
    X = np.load(os.path.join(PROC_DIR, f"{split}_{feature_type}{suffix}.npy"))
    y = np.load(os.path.join(PROC_DIR, f"{split}_labels{suffix}.npy"))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def compute_class_weights(y_train, n_classes):
    """Balanced inverse-frequency weights: n_samples / (n_classes * count).
    Counters the Normal-class imbalance (Normal recordings are much longer, so
    they yield far more windows) without discarding any data."""
    counts = np.bincount(y_train.numpy(), minlength=n_classes).astype(np.float64)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def train_one(feature_type, n_classes, train_suffix=""):
    X_train, y_train = load_split(feature_type, "train", suffix=train_suffix)
    X_val, y_val = load_split(feature_type, "val")
    X_test, y_test = load_split(feature_type, "test")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model = LightweightCNN1D(input_len=X_train.shape[1], n_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    class_weights = compute_class_weights(y_train, n_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  [{feature_type}] class weights: "
          f"{[round(w, 3) for w in class_weights.tolist()]}")

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(X_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [{feature_type}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    n_params = count_parameters(model)

    save_path = os.path.join(RESULTS_DIR, f"{feature_type}_model_fp32{train_suffix}.pt")
    torch.save(model.state_dict(), save_path)
    size_kb = os.path.getsize(save_path) / 1024

    print(f"[{feature_type}] FP32 -> acc={acc:.4f} f1={f1:.4f} "
          f"params={n_params} size={size_kb:.1f}KB")

    return {
        "feature_type": feature_type,
        "quantization": "FP32",
        "train_data": "aug" if train_suffix else "clean",
        "accuracy": acc,
        "f1_macro": f1,
        "n_params": n_params,
        "size_kb": size_kb,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aug", action="store_true",
                    help="train on noise-augmented features (run augment_train.py first)")
    args = ap.parse_args()
    suffix = "_aug" if args.aug else ""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_labels_path = os.path.join(PROC_DIR, f"train_labels{suffix}.npy")
    if not os.path.exists(train_labels_path):
        if args.aug:
            print("No augmented data - run src/augment_train.py first.")
        else:
            print("No processed data found - run src/load_data.py and src/features.py first.")
        return

    n_classes = len(np.unique(np.load(train_labels_path)))

    results = []
    for feature_type in ["fft", "wavelet"]:
        results.append(train_one(feature_type, n_classes, train_suffix=suffix))

    import csv
    log_path = os.path.join(RESULTS_DIR, f"train_log{suffix}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved training results to {log_path}")


if __name__ == "__main__":
    main()

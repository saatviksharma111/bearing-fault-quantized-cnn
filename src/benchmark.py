"""
Step 6: Consistent 4-way benchmark - {FFT, Wavelet} x {FP32, INT8}.
Measures accuracy, macro-F1, model size (KB), and average inference latency
on the SAME held-out test set for all four models.

Run: python src/benchmark.py
Output: results/benchmark_table.csv (your paper's centerpiece table)
"""
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from model import LightweightCNN1D

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

N_LATENCY_RUNS = 500
N_WARMUP = 20


def measure_latency(model, sample_input):
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(sample_input)
        times = []
        for _ in range(N_LATENCY_RUNS):
            start = time.perf_counter()
            model(sample_input)
            times.append(time.perf_counter() - start)
    return np.mean(times) * 1000, np.std(times) * 1000  # ms


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1).numpy()
    acc = accuracy_score(y_test.numpy(), preds)
    f1 = f1_score(y_test.numpy(), preds, average="macro")
    return acc, f1


def benchmark_one(feature_type, quant_type, n_classes, input_len, X_test, y_test):
    model = LightweightCNN1D(input_len=input_len, n_classes=n_classes)

    if quant_type == "FP32":
        state_path = os.path.join(RESULTS_DIR, f"{feature_type}_model_fp32.pt")
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
    else:  # INT8
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        state_path = os.path.join(RESULTS_DIR, f"{feature_type}_model_int8.pt")
        model.load_state_dict(torch.load(state_path, map_location="cpu"))

    size_kb = os.path.getsize(state_path) / 1024
    acc, f1 = evaluate_model(model, X_test, y_test)

    sample_input = X_test[0:1]
    latency_mean, latency_std = measure_latency(model, sample_input)

    return {
        "feature_type": feature_type,
        "quantization": quant_type,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "size_kb": round(size_kb, 2),
        "latency_ms_mean": round(latency_mean, 4),
        "latency_ms_std": round(latency_std, 4),
    }


def main():
    labels_path = os.path.join(PROC_DIR, "test_labels.npy")
    if not os.path.exists(labels_path):
        print("No processed data found - run the full pipeline first "
              "(load_data.py -> features.py -> train.py -> quantize.py).")
        return

    y_test_np = np.load(labels_path)
    n_classes = len(np.unique(y_test_np))
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    results = []
    for feature_type in ["fft", "wavelet"]:
        X_test_np = np.load(os.path.join(PROC_DIR, f"test_{feature_type}.npy"))
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        for quant_type in ["FP32", "INT8"]:
            model_path = os.path.join(RESULTS_DIR, f"{feature_type}_model_{quant_type.lower()}.pt")
            if not os.path.exists(model_path):
                print(f"Missing {model_path} - skipping. Run train.py / quantize.py first.")
                continue
            row = benchmark_one(feature_type, quant_type, n_classes, X_test_np.shape[1], X_test, y_test)
            results.append(row)
            print(row)

    if not results:
        print("No results produced - check that train.py and quantize.py ran successfully.")
        return

    out_path = os.path.join(RESULTS_DIR, "benchmark_table.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved 4-way benchmark table to {out_path}")
    print("This is your paper's centerpiece results table.")


if __name__ == "__main__":
    main()

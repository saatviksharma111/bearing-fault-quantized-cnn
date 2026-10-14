"""
Step 7: Noise robustness sweep.

Inject additive white Gaussian noise into the (time-domain) test windows at a
range of SNR levels, RE-EXTRACT FFT and wavelet features from the noisy signal,
and evaluate all four trained models (FFT/Wavelet x FP32/INT8) at each level.

Noise is added in the time domain *before* feature extraction - that is the
only way this test says anything about whether wavelet vs. FFT features are the
more noise-robust representation. Noise power is set per-window relative to that
window's signal power, so the requested SNR holds regardless of window energy.

Run: python src/noise_robustness.py
Output:
    results/noise_robustness.csv   - accuracy & macro-F1 for every (model, SNR)
    results/noise_robustness.png   - accuracy vs SNR, one line per model variant
"""
import os
import csv
import warnings

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from model import LightweightCNN1D
from features import extract_fft_features, extract_wavelet_features
from quantize import to_static_quantized

warnings.filterwarnings("ignore")

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# High-SNR points locate the degradation knee; the negative points (noise power
# exceeds signal power) are the stress test.
SNR_LEVELS_DB = [20, 12, 6, 0, -4, -8, -12]
N_TRIALS = 5                       # noise realizations to average per SNR
SEED = 42


def add_awgn(windows, snr_db, rng):
    """Add white Gaussian noise to hit `snr_db` per window.
    SNR_dB = 10*log10(P_signal / P_noise)  ->  P_noise = P_signal / 10^(SNR/10).
    """
    sig_power = np.mean(windows ** 2, axis=1, keepdims=True)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(windows.shape).astype(np.float32)
    noise *= np.sqrt(noise_power).astype(np.float32)
    return (windows + noise).astype(np.float32)


def features_from_windows(windows):
    fft = np.stack([extract_fft_features(w) for w in windows]).astype(np.float32)
    wav = np.stack([extract_wavelet_features(w) for w in windows]).astype(np.float32)
    return {"fft": fft, "wavelet": wav}


def load_model(feature_type, quant, input_len, n_classes, example_input):
    model = LightweightCNN1D(input_len=input_len, n_classes=n_classes)
    if quant == "FP32":
        path = os.path.join(RESULTS_DIR, f"{feature_type}_model_fp32.pt")
        model.load_state_dict(torch.load(path, map_location="cpu"))
    else:  # INT8 static - rebuild converted tree, then load calibrated params
        model = to_static_quantized(model, example_input=example_input)
        path = os.path.join(RESULTS_DIR, f"{feature_type}_model_int8.pt")
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def accuracy_of(model, X, y):
    preds = model(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()
    return accuracy_score(y, preds), f1_score(y, preds, average="macro")


def main():
    windows_path = os.path.join(PROC_DIR, "test_windows.npy")
    if not os.path.exists(windows_path):
        print("Missing test_windows.npy - run the pipeline through features.py first.")
        return

    test_windows = np.load(windows_path).astype(np.float32)
    y_test = np.load(os.path.join(PROC_DIR, "test_labels.npy"))
    n_classes = int(np.unique(y_test).size)
    input_len = 32  # feature-vector length (N_BANDS in features.py)

    # Clean features once - baseline row + priming input for the INT8 rebuild.
    clean_feats = features_from_windows(test_windows)

    variants = [(ft, q) for ft in ("fft", "wavelet") for q in ("FP32", "INT8")]
    models = {
        (ft, q): load_model(
            ft, q, input_len, n_classes,
            example_input=torch.tensor(clean_feats[ft][:32], dtype=torch.float32),
        )
        for ft, q in variants
    }

    rng = np.random.default_rng(SEED)
    rows = []

    # --- clean baseline (SNR = +inf) ---
    for (ft, q), model in models.items():
        acc, f1 = accuracy_of(model, clean_feats[ft], y_test)
        rows.append({"feature_type": ft, "quantization": q, "snr_db": "clean",
                     "accuracy": round(acc, 4), "f1_macro": round(f1, 4)})

    # --- noisy sweep, averaged over N_TRIALS realizations ---
    for snr in SNR_LEVELS_DB:
        acc_acc = {v: [] for v in variants}
        f1_acc = {v: [] for v in variants}
        for _ in range(N_TRIALS):
            noisy = add_awgn(test_windows, snr, rng)
            feats = features_from_windows(noisy)
            for v in variants:
                a, f = accuracy_of(models[v], feats[v[0]], y_test)
                acc_acc[v].append(a)
                f1_acc[v].append(f)
        for v in variants:
            rows.append({
                "feature_type": v[0], "quantization": v[1], "snr_db": snr,
                "accuracy": round(float(np.mean(acc_acc[v])), 4),
                "f1_macro": round(float(np.mean(f1_acc[v])), 4),
            })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "noise_robustness.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    _print_table(rows)
    _plot(rows)
    print(f"\nSaved {csv_path}")
    print(f"Saved {os.path.join(RESULTS_DIR, 'noise_robustness.png')}")


def _print_table(rows):
    order = ["clean"] + SNR_LEVELS_DB
    variants = [("fft", "FP32"), ("fft", "INT8"), ("wavelet", "FP32"), ("wavelet", "INT8")]
    lookup = {(r["feature_type"], r["quantization"], r["snr_db"]): r["accuracy"] for r in rows}
    header = f"{'model':<16}" + "".join(f"{('Clean' if s=='clean' else str(s)+'dB'):>9}" for s in order)
    print("\nAccuracy vs SNR (avg of "
          f"{N_TRIALS} noise realizations per level)\n" + "-" * len(header))
    print(header)
    for ft, q in variants:
        line = f"{ft + '-' + q:<16}"
        for s in order:
            line += f"{lookup[(ft, q, s)]:>9.4f}"
        print(line)


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = [("fft", "FP32"), ("fft", "INT8"), ("wavelet", "FP32"), ("wavelet", "INT8")]
    styles = {
        ("fft", "FP32"): dict(color="#c44", marker="o", ls="-"),
        ("fft", "INT8"): dict(color="#c44", marker="s", ls="--"),
        ("wavelet", "FP32"): dict(color="#268", marker="o", ls="-"),
        ("wavelet", "INT8"): dict(color="#268", marker="s", ls="--"),
    }
    # x-axis: place "Clean" a step to the left of the highest SNR
    step = SNR_LEVELS_DB[0] - SNR_LEVELS_DB[1]
    xs_num = [SNR_LEVELS_DB[0] + step] + SNR_LEVELS_DB
    xlabels = ["Clean"] + [str(s) for s in SNR_LEVELS_DB]
    snr_key = ["clean"] + SNR_LEVELS_DB
    lookup = {(r["feature_type"], r["quantization"], r["snr_db"]): r["accuracy"] for r in rows}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ft, q in variants:
        ys = [lookup[(ft, q, s)] for s in snr_key]
        ax.plot(xs_num, ys, label=f"{ft.upper()} {q}", **styles[(ft, q)])
    ax.set_xticks(xs_num)
    ax.set_xticklabels(xlabels)
    ax.invert_xaxis()  # noisier to the right
    ax.set_xlabel("SNR (dB)  -  noisier ->")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Noise robustness: FFT vs Wavelet features")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "noise_robustness.png"), dpi=130)


if __name__ == "__main__":
    main()

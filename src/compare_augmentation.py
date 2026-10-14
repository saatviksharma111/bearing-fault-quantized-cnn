"""
Step 7c: Does noise-augmented training push the collapse point lower?

Runs the identical noise-robustness sweep (Clean, then 20 dB down to -12 dB) on
BOTH model families:
    * clean-trained  : results/{fft,wavelet}_model_{fp32,int8}.pt
    * noise-augmented : results/{fft,wavelet}_model_{fp32,int8}_aug.pt
and writes one combined table so the two curves sit side by side.

Prereqs:
    python src/augment_train.py
    python src/train.py --aug
    python src/quantize.py --aug

Run: python src/compare_augmentation.py
Output:
    results/augmentation_comparison.csv   - accuracy at every (model, SNR)
    results/augmentation_comparison.png   - clean-trained vs augmented curves
"""
import csv
import os
import warnings

import numpy as np
import torch

from model import LightweightCNN1D
from quantize import to_static_quantized
from noise_robustness import (
    add_awgn, features_from_windows, accuracy_of, SNR_LEVELS_DB, N_TRIALS, SEED,
)

warnings.filterwarnings("ignore")

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

SNR_SWEEP = ["clean"] + SNR_LEVELS_DB          # ["clean", 20, 12, 6, 0, -4, -8, -12]
ROBUST_THRESHOLD = 0.90                         # "still working" cutoff
FEATURES = ["fft", "wavelet"]
QUANTS = ["FP32", "INT8"]
TRAININGS = ["clean", "aug"]                    # "" and "_aug" model suffixes


def load_model(feature_type, quant, training, input_len, n_classes, example_input):
    suffix = "" if training == "clean" else "_aug"
    model = LightweightCNN1D(input_len=input_len, n_classes=n_classes)
    if quant == "FP32":
        path = os.path.join(RESULTS_DIR, f"{feature_type}_model_fp32{suffix}.pt")
        model.load_state_dict(torch.load(path, map_location="cpu"))
    else:  # INT8 static - rebuild converted tree, then load calibrated params
        model = to_static_quantized(model, example_input=example_input)
        path = os.path.join(RESULTS_DIR, f"{feature_type}_model_int8{suffix}.pt")
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def robust_floor(acc_by_snr):
    """Lowest SNR in the sweep at which accuracy still >= ROBUST_THRESHOLD
    (contiguously from the clean end). Returns a label like '6 dB', 'clean', or
    'none'."""
    floor = None
    for snr in SNR_SWEEP:
        if acc_by_snr[snr] >= ROBUST_THRESHOLD:
            floor = snr
        else:
            break
    if floor is None:
        return "none"
    return "clean" if floor == "clean" else f"{floor} dB"


def main():
    win_path = os.path.join(PROC_DIR, "test_windows.npy")
    if not os.path.exists(win_path):
        print("Missing test_windows.npy - run the pipeline through features.py first.")
        return

    missing = [
        f"{ft}_model_{q.lower()}_aug.pt"
        for ft in FEATURES for q in QUANTS
        if not os.path.exists(os.path.join(RESULTS_DIR, f"{ft}_model_{q.lower()}_aug.pt"))
    ]
    if missing:
        print("Missing augmented models: " + ", ".join(missing))
        print("Run: python src/augment_train.py && python src/train.py --aug "
              "&& python src/quantize.py --aug")
        return

    test_windows = np.load(win_path).astype(np.float32)
    y_test = np.load(os.path.join(PROC_DIR, "test_labels.npy"))
    n_classes = int(np.unique(y_test).size)
    input_len = 32

    clean_feats = features_from_windows(test_windows)

    variants = [(ft, q, tr) for ft in FEATURES for q in QUANTS for tr in TRAININGS]
    models = {
        (ft, q, tr): load_model(
            ft, q, tr, input_len, n_classes,
            example_input=torch.tensor(clean_feats[ft][:32], dtype=torch.float32),
        )
        for ft, q, tr in variants
    }

    # acc[variant][snr] = mean accuracy over N_TRIALS realizations
    acc = {v: {} for v in variants}

    # clean baseline
    for v in variants:
        a, _ = accuracy_of(models[v], clean_feats[v[0]], y_test)
        acc[v]["clean"] = a

    rng = np.random.default_rng(SEED)
    for snr in SNR_LEVELS_DB:
        sums = {v: 0.0 for v in variants}
        for _ in range(N_TRIALS):
            noisy = add_awgn(test_windows, snr, rng)
            feats = features_from_windows(noisy)
            for v in variants:
                a, _ = accuracy_of(models[v], feats[v[0]], y_test)
                sums[v] += a
        for v in variants:
            acc[v][snr] = sums[v] / N_TRIALS

    _write_csv(acc, variants)
    _print_table(acc, variants)
    _plot(acc)
    print(f"\nSaved {os.path.join(RESULTS_DIR, 'augmentation_comparison.csv')}")
    print(f"Saved {os.path.join(RESULTS_DIR, 'augmentation_comparison.png')}")


def _row_label(ft, q, tr):
    return f"{ft.upper()}-{q}-{'clean-train' if tr == 'clean' else 'aug-train'}"


def _write_csv(acc, variants):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "augmentation_comparison.csv")
    cols = ["Clean" if s == "clean" else f"{s} dB" for s in SNR_SWEEP]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + cols + [f"robust_floor(acc>={ROBUST_THRESHOLD:.2f})"])
        for ft, q, tr in variants:
            row = [f"{acc[(ft, q, tr)][s]:.4f}" for s in SNR_SWEEP]
            w.writerow([_row_label(ft, q, tr)] + row + [robust_floor(acc[(ft, q, tr)])])


def _print_table(acc, variants):
    cols = ["Clean" if s == "clean" else f"{s}dB" for s in SNR_SWEEP]
    head = f"{'model':<26}" + "".join(f"{c:>9}" for c in cols) + f"{'floor':>9}"
    print(f"\nAccuracy vs SNR  (mean of {N_TRIALS} noise realizations)\n" + "-" * len(head))
    print(head)
    for ft, q, tr in variants:
        line = f"{_row_label(ft, q, tr):<26}"
        line += "".join(f"{acc[(ft, q, tr)][s]:>9.4f}" for s in SNR_SWEEP)
        line += f"{robust_floor(acc[(ft, q, tr)]):>9}"
        print(line)


def _plot(acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 200})

    numeric = SNR_LEVELS_DB
    step = numeric[0] - numeric[1]
    xs = [numeric[0] + step] + numeric
    xlabels = ["Clean"] + [str(s) for s in numeric]

    colors = {"clean": "#8E8E8E", "aug": "#1F6F8B"}
    lstyle = {"FP32": "-", "INT8": "--"}
    marker = {"FP32": "o", "INT8": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, ft in zip(axes, FEATURES):
        ax.axhline(0.40, color="0.6", ls=":", lw=1)
        ax.axhline(ROBUST_THRESHOLD, color="#C0392B", ls=":", lw=1, alpha=0.6)
        for q in QUANTS:
            for tr in TRAININGS:
                ys = [acc[(ft, q, tr)][s] for s in SNR_SWEEP]
                ax.plot(xs, ys, color=colors[tr], ls=lstyle[q], marker=marker[q],
                        ms=5, lw=1.9,
                        label=f"{q}, {'clean-train' if tr == 'clean' else 'aug-train'}")
        ax.invert_xaxis()
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Test-set SNR (dB)  -  noisier ->")
        ax.set_title(f"{ft.upper()} features")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test accuracy (4-class)")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    fig.suptitle("Noise-augmented vs clean training  -  does augmentation lower the collapse point?",
                 x=0.5, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "augmentation_comparison.png"))
    fig.savefig(os.path.join(RESULTS_DIR, "augmentation_comparison.pdf"))


if __name__ == "__main__":
    main()

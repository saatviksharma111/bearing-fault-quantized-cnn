"""
Step 1-2: Load CWRU .mat files, split by FILE (not by window) into
train/val/test, then window each split.

Run: python src/load_data.py
Output: data/processed/{train,val,test}_windows.npy and *_labels.npy
"""
import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from labels import FILENAME_TO_LABEL, find_de_signal_key

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

WINDOW_SIZE = 1024
OVERLAP = 0.5  # 50% overlap
RANDOM_SEED = 42


def load_signal(filepath):
    mat = loadmat(filepath)
    key = find_de_signal_key(mat)
    signal = mat[key].squeeze().astype(np.float32)
    return signal


def window_signal(signal, window_size=WINDOW_SIZE, overlap=OVERLAP):
    step = int(window_size * (1 - overlap))
    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[start : start + window_size])
    return np.array(windows, dtype=np.float32)


def normalize_windows(windows, mean=None, std=None):
    """Z-score normalize. Pass train-set mean/std to apply the SAME
    normalization to val/test (never compute normalization stats on val/test
    directly — that's a subtle form of leakage too)."""
    if mean is None or std is None:
        mean = windows.mean()
        std = windows.std()
    return (windows - mean) / (std + 1e-8), mean, std


def main():
    if not FILENAME_TO_LABEL:
        print(
            "FILENAME_TO_LABEL in src/labels.py is empty.\n"
            "Download the CWRU .mat files, put them in data/raw/, and fill in\n"
            "the filename -> label mapping in src/labels.py before running this."
        )
        return

    os.makedirs(PROC_DIR, exist_ok=True)

    filenames = list(FILENAME_TO_LABEL.keys())
    labels = [FILENAME_TO_LABEL[f] for f in filenames]

    # Split by FILE first — this is the leak-proof step.
    # test_size=0.5 so temp keeps >=2 files per class (only 4 files/class total),
    # which the val/test stratified split below requires.
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        filenames, labels, test_size=0.5, stratify=labels, random_state=RANDOM_SEED
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=RANDOM_SEED
    )

    splits = {
        "train": (train_files, train_labels),
        "val": (val_files, val_labels),
        "test": (test_files, test_labels),
    }

    train_mean, train_std = None, None

    for split_name, (files, file_labels) in splits.items():
        all_windows = []
        all_labels = []
        for fname, label in zip(files, file_labels):
            filepath = os.path.join(RAW_DIR, fname)
            signal = load_signal(filepath)
            windows = window_signal(signal)
            all_windows.append(windows)
            all_labels.extend([label] * len(windows))

        all_windows = np.concatenate(all_windows, axis=0)
        all_labels = np.array(all_labels, dtype=np.int64)

        if split_name == "train":
            all_windows, train_mean, train_std = normalize_windows(all_windows)
        else:
            all_windows, _, _ = normalize_windows(all_windows, train_mean, train_std)

        np.save(os.path.join(PROC_DIR, f"{split_name}_windows.npy"), all_windows)
        np.save(os.path.join(PROC_DIR, f"{split_name}_labels.npy"), all_labels)

        print(f"{split_name}: {all_windows.shape[0]} windows, "
              f"class distribution: {np.bincount(all_labels)}")

    print(f"\nSaved train mean={train_mean:.4f}, std={train_std:.4f} "
          f"(reused for val/test normalization)")


if __name__ == "__main__":
    main()

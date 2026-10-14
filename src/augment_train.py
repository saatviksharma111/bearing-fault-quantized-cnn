"""
Step 4b (prep): Build a noise-AUGMENTED training set.

Takes the clean training windows and, before any feature extraction, adds white
Gaussian noise at a per-window random SNR drawn uniformly from [SNR_MIN, SNR_MAX]
dB. The clean windows are kept as well (so clean accuracy does not regress), and
AUG_COPIES independent noisy replicas are appended. FFT and wavelet features are
then extracted from the combined set exactly as in features.py.

Run: python src/augment_train.py
Output:
    data/processed/train_fft_aug.npy
    data/processed/train_wavelet_aug.npy
    data/processed/train_labels_aug.npy

Then:  python src/train.py --aug   and   python src/quantize.py --aug
"""
import os
import numpy as np

from features import extract_fft_features, extract_wavelet_features

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

SNR_MIN_DB = 0
SNR_MAX_DB = 20
AUG_COPIES = 4      # noisy replicas appended to the clean windows
SEED = 1234


def add_awgn_random_snr(windows, snr_min, snr_max, rng):
    """One independent noise draw per window, each at its own uniform-random SNR."""
    snr_db = rng.uniform(snr_min, snr_max, size=(windows.shape[0], 1)).astype(np.float32)
    sig_power = np.mean(windows ** 2, axis=1, keepdims=True)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(windows.shape).astype(np.float32)
    noise *= np.sqrt(noise_power).astype(np.float32)
    return (windows + noise).astype(np.float32)


def main():
    win_path = os.path.join(PROC_DIR, "train_windows.npy")
    if not os.path.exists(win_path):
        print("Missing train_windows.npy - run src/load_data.py first.")
        return

    clean_w = np.load(win_path).astype(np.float32)
    clean_y = np.load(os.path.join(PROC_DIR, "train_labels.npy"))
    rng = np.random.default_rng(SEED)

    parts_w = [clean_w]
    parts_y = [clean_y]
    for _ in range(AUG_COPIES):
        parts_w.append(add_awgn_random_snr(clean_w, SNR_MIN_DB, SNR_MAX_DB, rng))
        parts_y.append(clean_y)

    aug_w = np.concatenate(parts_w, axis=0)
    aug_y = np.concatenate(parts_y, axis=0).astype(np.int64)

    perm = rng.permutation(len(aug_w))
    aug_w, aug_y = aug_w[perm], aug_y[perm]

    print(f"augmented train windows: {aug_w.shape} "
          f"(clean {len(clean_w)} + {AUG_COPIES} x noisy, "
          f"SNR ~ U[{SNR_MIN_DB}, {SNR_MAX_DB}] dB)")
    print(f"class distribution: {np.bincount(aug_y)}")

    fft = np.stack([extract_fft_features(w) for w in aug_w]).astype(np.float32)
    wav = np.stack([extract_wavelet_features(w) for w in aug_w]).astype(np.float32)

    np.save(os.path.join(PROC_DIR, "train_fft_aug.npy"), fft)
    np.save(os.path.join(PROC_DIR, "train_wavelet_aug.npy"), wav)
    np.save(os.path.join(PROC_DIR, "train_labels_aug.npy"), aug_y)

    print(f"saved train_fft_aug {fft.shape}, train_wavelet_aug {wav.shape}, "
          f"train_labels_aug {aug_y.shape}")


if __name__ == "__main__":
    main()

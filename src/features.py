"""
Step 3: Two parallel feature extraction pipelines - FFT-based and
Wavelet-based - kept at comparable dimensionality for a fair comparison.

Run: python src/features.py
Output: data/processed/{split}_{fft,wavelet}.npy for split in train/val/test
"""
import os
import numpy as np
import pywt

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

N_BANDS = 32          # target feature dimensionality for both pipelines
WAVELET = "db4"        # Daubechies-4, common choice for vibration signals
WAVELET_DEPTH = 5      # decomposition depth -> 2^5 = 32 sub-bands, matches N_BANDS


def extract_fft_features(window, n_bands=N_BANDS):
    """Magnitude spectrum binned into n_bands energy bands."""
    spectrum = np.abs(np.fft.rfft(window))
    # drop DC component, split remainder into n_bands equal chunks, sum energy
    spectrum = spectrum[1:]
    band_edges = np.linspace(0, len(spectrum), n_bands + 1).astype(int)
    features = np.array(
        [np.sum(spectrum[band_edges[i]:band_edges[i + 1]] ** 2) for i in range(n_bands)],
        dtype=np.float32,
    )
    # log-scale energy (common practice, keeps large/small values comparable)
    return np.log1p(features)


def extract_wavelet_features(window, wavelet=WAVELET, depth=WAVELET_DEPTH):
    """Wavelet packet decomposition -> energy per terminal sub-band."""
    wp = pywt.WaveletPacket(data=window, wavelet=wavelet, mode="symmetric", maxlevel=depth)
    nodes = [node.path for node in wp.get_level(depth, order="natural")]
    features = np.array(
        [np.sum(np.array(wp[node].data) ** 2) for node in nodes], dtype=np.float32
    )
    return np.log1p(features)


def extract_statistical_features(window):
    """Optional cheap baseline: RMS, kurtosis, crest factor, skewness."""
    from scipy.stats import kurtosis, skew
    rms = np.sqrt(np.mean(window ** 2))
    peak = np.max(np.abs(window))
    crest = peak / (rms + 1e-8)
    return np.array([rms, kurtosis(window), crest, skew(window)], dtype=np.float32)


def process_split(split_name):
    windows = np.load(os.path.join(PROC_DIR, f"{split_name}_windows.npy"))

    fft_feats = np.array([extract_fft_features(w) for w in windows])
    wavelet_feats = np.array([extract_wavelet_features(w) for w in windows])

    np.save(os.path.join(PROC_DIR, f"{split_name}_fft.npy"), fft_feats)
    np.save(os.path.join(PROC_DIR, f"{split_name}_wavelet.npy"), wavelet_feats)

    print(f"{split_name}: fft_feats {fft_feats.shape}, wavelet_feats {wavelet_feats.shape}")


def main():
    for split in ["train", "val", "test"]:
        windows_path = os.path.join(PROC_DIR, f"{split}_windows.npy")
        if not os.path.exists(windows_path):
            print(f"Missing {windows_path} - run src/load_data.py first.")
            return
        process_split(split)


if __name__ == "__main__":
    main()

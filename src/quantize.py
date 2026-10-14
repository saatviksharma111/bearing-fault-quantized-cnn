"""
Step 5: Post-training STATIC quantization of both trained models (FFT-CNN,
Wavelet-CNN) from FP32 to INT8.

Static quantization (unlike dynamic) also quantizes the Conv1d layers, which
hold ~97% of this model's parameters and virtually all of its compute. The
recipe is:
    1. Load the FP32 model, put it in eval() mode.
    2. Fuse Conv+BN+ReLU triplets (model.fuse_model()) so BatchNorm is folded
       into the Conv weights.
    3. Attach a qconfig and prepare() -> inserts activation observers.
    4. Calibrate: run a few hundred representative TRAINING samples through the
       prepared model so the observers learn realistic activation ranges.
    5. convert() -> swaps modules for their INT8 quantized implementations.

Run: python src/quantize.py
Output: results/{fft,wavelet}_model_int8.pt
"""
import argparse
import os
import warnings

import numpy as np
import torch

from model import LightweightCNN1D

warnings.filterwarnings("ignore")  # torch.ao.quantization deprecation noise

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

QCONFIG_BACKEND = "onednn"   # CPU backend available in this torch build
N_CALIBRATION = 512       # training samples to estimate activation ranges
CALIB_BATCH = 32


def to_static_quantized(model, calibration_batches=None, example_input=None):
    """Fuse -> prepare -> (calibrate) -> convert. Returns an INT8 model.

    Pass `calibration_batches` (an iterable of input tensors) to actually
    calibrate. `example_input` is a fallback single batch used only to give the
    observers a finite range when rebuilding the architecture for loading a
    saved INT8 state_dict (benchmark.py) - the loaded state_dict then overwrites
    every scale / zero-point / packed weight anyway.
    """
    torch.backends.quantized.engine = QCONFIG_BACKEND
    model.eval()
    model.fuse_model()
    model.qconfig = torch.ao.quantization.get_default_qconfig(QCONFIG_BACKEND)
    torch.ao.quantization.prepare(model, inplace=True)

    with torch.no_grad():
        if calibration_batches is not None:
            for xb in calibration_batches:
                model(xb)
        elif example_input is not None:
            model(example_input)

    torch.ao.quantization.convert(model, inplace=True)
    return model


def _calibration_batches(feature_type, suffix=""):
    X = np.load(os.path.join(PROC_DIR, f"train_{feature_type}{suffix}.npy"))
    Xt = torch.tensor(X[:N_CALIBRATION], dtype=torch.float32)
    return [Xt[i : i + CALIB_BATCH] for i in range(0, len(Xt), CALIB_BATCH)]


def quantize_static(feature_type, n_classes, input_len, suffix=""):
    model = LightweightCNN1D(input_len=input_len, n_classes=n_classes)
    state_path = os.path.join(RESULTS_DIR, f"{feature_type}_model_fp32{suffix}.pt")
    model.load_state_dict(torch.load(state_path, map_location="cpu"))

    model = to_static_quantized(
        model, calibration_batches=_calibration_batches(feature_type, suffix)
    )

    save_path = os.path.join(RESULTS_DIR, f"{feature_type}_model_int8{suffix}.pt")
    torch.save(model.state_dict(), save_path)
    size_kb = os.path.getsize(save_path) / 1024
    fp32_kb = os.path.getsize(state_path) / 1024
    print(
        f"[{feature_type}{suffix}] INT8 (static) saved -> {size_kb:.1f}KB "
        f"(FP32 {fp32_kb:.1f}KB, {fp32_kb / size_kb:.2f}x smaller)"
    )
    return model


def main():
    ap = argparse.ArgumentParser(description="Static INT8 quantization of the trained CNNs.")
    ap.add_argument("--aug", action="store_true",
                    help="quantize the noise-augmented models (*_fp32_aug.pt)")
    args = ap.parse_args()
    suffix = "_aug" if args.aug else ""

    os.makedirs(RESULTS_DIR, exist_ok=True)
    labels_path = os.path.join(PROC_DIR, f"train_labels{suffix}.npy")
    if not os.path.exists(labels_path):
        print("No processed data found - run the pipeline "
              f"({'augment_train.py' if args.aug else 'load_data.py -> features.py'}) first.")
        return
    n_classes = len(np.unique(np.load(labels_path)))

    for feature_type in ["fft", "wavelet"]:
        fp32_path = os.path.join(RESULTS_DIR, f"{feature_type}_model_fp32{suffix}.pt")
        if not os.path.exists(fp32_path):
            print(f"Missing {fp32_path} - run src/train.py{' --aug' if args.aug else ''} first.")
            continue
        X = np.load(os.path.join(PROC_DIR, f"train_{feature_type}{suffix}.npy"))
        quantize_static(feature_type, n_classes, input_len=X.shape[1], suffix=suffix)


if __name__ == "__main__":
    main()

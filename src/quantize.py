"""
Step 5: Post-training quantization of both trained models (FFT-CNN,
Wavelet-CNN) from FP32 to INT8.

Uses dynamic quantization (torch.quantization.quantize_dynamic) as the
starting point - it's the simplest to get working, quantizing Linear layers'
weights to INT8 at runtime. NOTE: dynamic quantization mainly benefits
Linear/LSTM layers; Conv1d layers are NOT quantized by quantize_dynamic in
standard PyTorch. For full Conv1d quantization you need STATIC quantization
(prepare/calibrate/convert) - see the commented-out section below as the
next step once dynamic quantization is working end-to-end.

Run: python src/quantize.py
Output: results/{fft,wavelet}_model_int8.pt
"""
import os
import torch
import torch.nn as nn
import numpy as np

from model import LightweightCNN1D

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def quantize_dynamic(feature_type, n_classes, input_len):
    model = LightweightCNN1D(input_len=input_len, n_classes=n_classes)
    state_path = os.path.join(RESULTS_DIR, f"{feature_type}_model_fp32.pt")
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    # Dynamic quantization: quantizes Linear layer weights to INT8.
    # Simplest starting point - upgrade to static quantization (see note
    # below) once this pipeline runs end-to-end, since static quantization
    # also quantizes the Conv1d layers and gives bigger size/speed wins.
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    save_path = os.path.join(RESULTS_DIR, f"{feature_type}_model_int8.pt")
    torch.save(quantized_model.state_dict(), save_path)
    size_kb = os.path.getsize(save_path) / 1024
    print(f"[{feature_type}] INT8 (dynamic) saved -> {size_kb:.1f}KB")
    return quantized_model


# --- Next step once dynamic quantization works (static quantization) ---
# Static quantization also quantizes Conv1d layers, which is where most of
# your model's compute lives, so it gives a much more meaningful size/speed
# comparison. It requires: (1) fusing Conv+BatchNorm+ReLU, (2) a calibration
# pass over representative data, (3) convert. Sketch:
#
# model.eval()
# model_fused = torch.quantization.fuse_modules(
#     model, [["features.0", "features.1", "features.2"], ...]  # match your layer names
# )
# model_fused.qconfig = torch.quantization.get_default_qconfig("fbgemm")
# model_prepared = torch.quantization.prepare(model_fused)
# with torch.no_grad():
#     for xb, _ in calibration_loader:   # run a few hundred training samples through
#         model_prepared(xb)
# model_quantized = torch.quantization.convert(model_prepared)
#
# This is flagged in the project plan as the step most likely to need
# debugging time - start with dynamic quantization to get the full pipeline
# (train -> quantize -> benchmark) working, then swap in static quantization.


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    labels_path = os.path.join(PROC_DIR, "train_labels.npy")
    if not os.path.exists(labels_path):
        print("No processed data found - run src/load_data.py and src/features.py first.")
        return
    n_classes = len(np.unique(np.load(labels_path)))

    for feature_type in ["fft", "wavelet"]:
        fp32_path = os.path.join(RESULTS_DIR, f"{feature_type}_model_fp32.pt")
        if not os.path.exists(fp32_path):
            print(f"Missing {fp32_path} - run src/train.py first.")
            continue
        X = np.load(os.path.join(PROC_DIR, f"train_{feature_type}.npy"))
        quantize_dynamic(feature_type, n_classes, input_len=X.shape[1])


if __name__ == "__main__":
    main()

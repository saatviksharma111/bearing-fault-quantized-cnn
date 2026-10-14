# Lightweight Quantized 1D-CNN for Bearing Fault Detection

Course project (AIML): a small 1D-CNN that classifies bearing faults from
vibration data, comparing **FFT vs wavelet-packet features** and **FP32 vs INT8
quantization** for edge deployment.

> **Status:** `src/` currently holds the initial pipeline skeleton (random
> window split, dynamic quantization). The improved version (split by motor
> load, class-weighted loss, static INT8 quantization, noise-robustness sweep,
> noise-augmented training) will be added in a later commit.

## Dataset
Case Western Reserve University Bearing Data Center, 12 kHz drive-end data
(not included in this repo). Download from
https://engineering.case.edu/bearingdatacenter/download-data-file and put the
`.mat` files in `data/raw/`.

Files used (0.007" fault diameter, loads 0-3 HP):

| Class | Label | Files |
|---|---|---|
| Normal | 0 | 97-100 |
| Inner Race | 1 | 105-108 |
| Outer Race (centered) | 2 | 130-133 |
| Ball | 3 | 118-121 |

## Structure
```
src/
  labels.py      filename -> label mapping
  load_data.py   load .mat, split, window, normalize
  features.py    FFT and wavelet-packet features
  model.py       lightweight 1D-CNN
  train.py       FP32 training
  quantize.py    post-training quantization
  benchmark.py   FFT/Wavelet x FP32/INT8 comparison
```

## Run
```bash
pip install torch PyWavelets scikit-learn numpy scipy matplotlib
python src/load_data.py
python src/features.py
python src/train.py
python src/quantize.py
python src/benchmark.py
```

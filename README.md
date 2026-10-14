# Lightweight Quantized 1D-CNN for Bearing Fault Detection

Course project (AIML): a small 1D-CNN that classifies bearing faults from
vibration data, comparing **FFT vs wavelet-packet features** and **FP32 vs INT8
quantization** for edge deployment.

## Results (CWRU, 0.007" faults, 4 classes)

| Features | Precision | Accuracy | Macro-F1 | Size (KB) |
|---|---|---|---|---|
| FFT | FP32 | 1.00 | 1.00 | 41.2 |
| FFT | INT8 | 1.00 | 1.00 | 18.6 |
| Wavelet | FP32 | 1.00 | 1.00 | 41.3 |
| Wavelet | INT8 | 1.00 | 1.00 | 18.7 |

Static INT8 quantization shrinks the model about 2.2x with no accuracy loss on
clean data. Single-sample CPU latency was worse for INT8 on a desktop CPU.

Noise robustness (white noise at a given SNR; `results/augmentation_comparison.csv`):
clean-trained models fail below about 12 dB (wavelet holds up better than FFT
at 12 dB). Training with noise augmentation (SNR sampled 0-20 dB) moves the
accuracy floor to about -4 dB for FFT and 0 dB for wavelet; the ordering of the
two feature types flips at very low SNR.

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
  load_data.py   load .mat, split by file, window, normalize
  features.py    FFT and wavelet-packet features
  model.py       lightweight 1D-CNN
  train.py       FP32 training
  quantize.py    post-training quantization
  benchmark.py   FFT/Wavelet x FP32/INT8 comparison
  noise_robustness.py     SNR sweep (clean-trained models)
  augment_train.py        noise-augmented training
  compare_augmentation.py clean vs augmented comparison
  make_report_figure.py   report figures
results/         CSV tables and figures (model weights not tracked)
```

## Run
```bash
pip install torch PyWavelets scikit-learn numpy scipy matplotlib
python src/load_data.py
python src/features.py
python src/train.py
python src/quantize.py
python src/benchmark.py
python src/noise_robustness.py
python src/augment_train.py
python src/compare_augmentation.py
```

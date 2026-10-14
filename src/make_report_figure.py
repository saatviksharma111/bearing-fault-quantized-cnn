"""
Step 7b: Turn the raw noise-robustness sweep (results/noise_robustness.csv,
long format) into report-ready artifacts:

    results/noise_robustness_report.png  - clean, annotated accuracy-vs-SNR plot
    results/noise_robustness_report.pdf  - vector version for the report
    results/noise_robustness_table.csv   - wide accuracy table (model x SNR)

Run: python src/make_report_figure.py
"""
import csv
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SRC_CSV = os.path.join(RESULTS_DIR, "noise_robustness.csv")

# SNR columns in display order; "clean" is the noise-free baseline.
SNR_ORDER = ["clean", 20, 12, 6, 0, -4, -8, -12]
VARIANTS = [("fft", "FP32"), ("fft", "INT8"), ("wavelet", "FP32"), ("wavelet", "INT8")]
BREAKDOWN_DB = 12  # last SNR where wavelet still holds while FFT has fallen


def load_rows():
    with open(SRC_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    acc = {}
    for r in rows:
        snr = r["snr_db"]
        snr = snr if snr == "clean" else int(snr)
        acc[(r["feature_type"], r["quantization"], snr)] = float(r["accuracy"])
    return acc


def write_wide_table(acc):
    out = os.path.join(RESULTS_DIR, "noise_robustness_table.csv")
    headers = ["model"] + [("Clean" if s == "clean" else f"{s} dB") for s in SNR_ORDER]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for ft, q in VARIANTS:
            w.writerow([f"{ft.upper()}-{q}"] + [f"{acc[(ft, q, s)]:.4f}" for s in SNR_ORDER])
    print(f"Saved {out}")
    return out


def make_figure(acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "figure.dpi": 200,
    })

    numeric_snr = [s for s in SNR_ORDER if s != "clean"]          # 20 .. -12
    step = numeric_snr[0] - numeric_snr[1]                        # 8
    x_clean = numeric_snr[0] + step                               # place "Clean" left of 20 dB
    xs = [x_clean] + numeric_snr
    xlabels = ["Clean"] + [str(s) for s in numeric_snr]

    styles = {
        ("fft", "FP32"):     dict(color="#C0392B", marker="o", ls="-",  lw=2),
        ("fft", "INT8"):     dict(color="#C0392B", marker="s", ls="--", lw=1.8, ms=5),
        ("wavelet", "FP32"): dict(color="#1F6F8B", marker="o", ls="-",  lw=2),
        ("wavelet", "INT8"): dict(color="#1F6F8B", marker="s", ls="--", lw=1.8, ms=5),
    }
    labels = {
        ("fft", "FP32"): "FFT features, FP32",
        ("fft", "INT8"): "FFT features, INT8",
        ("wavelet", "FP32"): "Wavelet features, FP32",
        ("wavelet", "INT8"): "Wavelet features, INT8",
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # chance / majority-class reference
    ax.axhline(0.40, color="0.55", ls=":", lw=1)
    ax.text(0, 0.43, "majority-class baseline (0.40)",
            color="0.4", fontsize=8.5, va="bottom", ha="center")

    for v in VARIANTS:
        ys = [acc[(v[0], v[1], s)] for s in SNR_ORDER]
        ax.plot(xs, ys, label=labels[v], **styles[v])

    # --- annotate the ~12 dB breakdown point ---
    fft_12 = acc[("fft", "FP32", BREAKDOWN_DB)]
    wav_12 = acc[("wavelet", "FP32", BREAKDOWN_DB)]
    ax.axvline(BREAKDOWN_DB, color="0.3", ls="-", lw=0.9, alpha=0.6)
    ax.annotate(
        f"~{BREAKDOWN_DB} dB breakdown:\n"
        f"wavelet {wav_12:.0%} vs FFT {fft_12:.0%}",
        xy=(BREAKDOWN_DB, fft_12),
        xytext=(BREAKDOWN_DB - 2.0, 0.50),
        ha="right", va="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="#FCF3CF", ec="0.5", lw=0.8),
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1),
    )
    # bracket the gap at 12 dB
    ax.annotate("", xy=(BREAKDOWN_DB, wav_12), xytext=(BREAKDOWN_DB, fft_12),
                arrowprops=dict(arrowstyle="<->", color="0.3", lw=1))

    ax.invert_xaxis()  # decreasing SNR (increasing noise) to the right
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Test-set SNR (dB) — decreasing signal-to-noise ratio →")
    ax.set_ylabel("Test accuracy (4-class)")
    ax.set_title("Noise robustness of FFT vs. wavelet features\n"
                 "Gaussian noise injected in the time domain; 5 realizations averaged",
                 loc="left")
    ax.set_ylim(0, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()

    png = os.path.join(RESULTS_DIR, "noise_robustness_report.png")
    pdf = os.path.join(RESULTS_DIR, "noise_robustness_report.pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


def main():
    if not os.path.exists(SRC_CSV):
        print(f"Missing {SRC_CSV} - run src/noise_robustness.py first.")
        return
    acc = load_rows()
    write_wide_table(acc)
    make_figure(acc)


if __name__ == "__main__":
    main()

# Noise2Noise and Noise2Clean denoising for time-resolved 1D SAXS curves

Training and inference code for two denoising networks that operate on **one-dimensional SAXS curves**
`I(q)` of variable length: a **self-supervised (Noise2Noise)** model that needs no clean reference, and a
**supervised (Noise2Clean)** model that serves as a benchmark.

---

## Contents

| File | Role |
|---|---|
| `N2N_curv_09_train.py` | Trains the self-supervised model (`Noise2Noise`) |
| `N2N_curv_09-16_test.py` | GUI for denoising curves with the trained Noise2Noise model and for scoring the result without a reference |
| `curve_smooth_NNAttention11_1.py` | Trains the supervised benchmark (`Noise2Clean`) |
| `curve_smooth_test11_oneGUI.py` | GUI for denoising curves with the trained Noise2Clean model at a user-set denoising factor |

The four scripts total about 4,200 lines: 1,404 (`N2N_curv_09-16_test.py`), 1,035
(`curve_smooth_test11_oneGUI.py`), 1,029 (`curve_smooth_NNAttention11_1.py`) and 750
(`N2N_curv_09_train.py`).

---

## Requirements

Python 3, with

```
torch                  (tested with 2.5.1; CUDA optional)
numpy
pandas
matplotlib
scikit-learn
chardet                (only for N2N_curv_09_train.py)
tkinter                (only for the two GUI tools; ships with CPython on Windows/macOS,
                        on Linux install python3-tk)
```

A GPU is not required, but training the full datasets takes about 2 h (Noise2Clean) and 1 h
(Noise2Noise) on an NVIDIA RTX 3060. Inference on a single curve takes < 0.1 s on CPU.

---

## Input data format

All four scripts read plain-text, whitespace-separated column files, and all of them skip comment
lines. `N2N_curv_09_train.py` detects the file encoding with `chardet`; `N2N_curv_09-16_test.py`
tries the detected encoding followed by UTF-8 and GBK; `curve_smooth_test11_oneGUI.py` tries UTF-8,
latin-1, cp1252 and iso-8859-1 in turn. `curve_smooth_NNAttention11_1.py` assumes UTF-8 (via
`pandas.read_csv`) — convert your files to UTF-8 if they are not already.

**Noise2Noise** (`N2N_curv_09_train.py`, `N2N_curv_09-16_test.py`) — two or more columns, of which the
first is `q` and the second is the intensity:

```
# q(A^-1)    I(a.u.)
0.0100       1.234
0.0105       1.198
...
```

**Noise2Clean** (`curve_smooth_NNAttention11_1.py`) — four or more columns, taken from the paired
`.dat` files written by the pairing script: the first column is the scattering vector, the second the
noisy intensity, the third the scattering vector again, and the fourth the smooth reference intensity
of the same curve.

```
# Data from /raw/sample-name/series/curve-0001.dat
# Smooth curve from /raw/sample-name/series/curve-last.dat
# SmoothFactor: 16.0
# Column1: q_data, Column2: I_data, Column3: q_smo, Column4: I_smo
0.01000  1.234000  0.01000  1.241000
0.01050  1.198000  0.01050  1.205000
...
```

Two header lines matter to the code. `# Data from` names the file the noisy curve was cut from; the
folder it sits in identifies the **sample**, which is the unit the dataset is split on. `# SmoothFactor:`
records the averaging factor `N` used to build the smooth reference and is passed to the network as a
conditioning input; if the line is absent, 16.0 is assumed.

---

## Usage

### 1. Self-supervised model (Noise2Noise)

**Training.** Put the curve files into subfolders of `./Data/`, **one subfolder per sample** (the 50
consecutive frames of a static sample belong in one subfolder — training pairs are drawn at random
from within a subfolder). The training / validation / test split is made **on the sample folders**,
not on the pairs, so every frame of a given sample stays inside a single subset and the held-out
loss is measured on samples the model has never seen. Then run

```bash
python N2N_curv_09_train.py
```

Key settings are at the top of the file, in `class Config`:

| Setting | Default value |
|---|---|
| `num_pairs` | 5400 |
| `batch_size` | 1 (curves have variable length and are processed one at a time) |
| `num_epochs` | 500 |
| `learning_rate` | 1e-3 (Adam), cosine annealing to 1e-5 |
| split (train : val : test) | 70 : 15 : 15, fixed seed 42 |
| loss | L1 |
| early stopping | patience 100, min_delta 1e-6 |
| preprocessing | `clip_negative = True`, `use_log1p = True` |
| random seed | 42 |

Outputs `./models/denoise_1d.pth` (best validation loss) and `./models/denoise_1d_final.pth`.

**Inference and scoring.** Run

```bash
python N2N_curv_09-16_test.py
```

and select a curve file in the GUI. The tool loads `./models/denoise_1d.pth`, denoises the curve, and
reports:

- first- and second-order difference norms (L1 and L2) of the raw and denoised curve, and their reductions;
- the **balance ratio** *R*, defined as the reduction in second-order norm divided by the reduction in
  first-order norm. *R* > 1 means noise-like curvature is removed faster than signal-bearing gradient.
  This is the reference-free metric: it needs no clean reference measurement;
- the autocorrelation of the residuals, which should lie inside the 95 % confidence band at every
  non-zero lag.

The denoised curve can be exported to `.dat`/`.txt` with the metric values written into the header, and
the figure to `.svg` at 600 dpi.

### 2. Supervised benchmark (Noise2Clean)

**Training.** Put the paired curves into `./curve_pairs/` in the four-column format above, then run

```bash
python curve_smooth_NNAttention11_1.py
```

Settings are at the top of the `if __name__ == "__main__":` block: `folder_path` (default
`./curve_pairs`) and `use_log_coordinates` (default `True`). The split is again made **on the samples,
not on the curve pairs**: the sample a pair belongs to is read back from the `# Data from <path>`
header that the pairing script writes into each file, so all the pairs cut from one sample stay in a
single subset. The ratio is 70 : 15 : 15 with seed 42. Training uses batch size 32, 500 epochs, Adam
at 5e-3 with weight decay 1e-5, `ReduceLROnPlateau` (factor 0.5, patience 5), and early stopping with
patience 10. The objective combines mean-square and mean-absolute error with smoothness,
attention-sparsity, peak-protection, edge and flat-region terms.

Outputs `best_smoothing_model_NN11_v2.pth` and `final_smoothing_model_NN11_v2.pth`.

**Inference.** Run

```bash
python curve_smooth_test11_oneGUI.py
```

select a curve file, set the **denoising factor** (smooth factor), and export the plot. The network
predicts spatially varying denoising weights, so a single trained model can be asked for different
amounts of smoothing; it can also be applied iteratively to the same curve.

---

## Notes on the two paradigms

- **Noise2Noise** requires only that the noise of two frames of the same static sample be zero-mean and
  independent between the frames. It does **not** require the noise to follow any particular
  distribution — the Poisson character is checked here as a convenient, demanding test of
  those two conditions, not as a precondition of the method.
- **Noise2Clean** exists in this repository only to provide the supervised upper bound against which
  the self-supervised result is compared. A genuinely clean reference cannot be measured in a
  time-resolved experiment; the smooth target is approximated by averaging consecutive frames of a
  static sample, which requires the sample to stay unchanged over the acquisition.

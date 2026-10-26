# Training and reproducibility

An environmental sound classification portfolio project: upload a short recording and explore the model's top five sound labels.

[Try the live demo](https://soundsight-ran-nope.streamlit.app/) · [Free deployment instructions](DEPLOYMENT.md)

**Measured first baseline:** 57.55% mean held-out accuracy and 0.5626 mean macro F1 across five source-isolated folds. See [the evaluation report](RESULTS.md) for the exact protocol, fold results and limitations. Sixteen software tests pass, including decoding real MP3, FLAC, OGG and AIFF files. These results describe ESC-50 evaluation, not accuracy on arbitrary uploads.

## MVP

Python audio preprocessing → fixed acoustic features → a trained scikit-learn classifier → honest held-out evaluation → Streamlit upload-and-predict demo. This is a CPU-friendly classical ML baseline, with no pretrained network or external inference API.

The [official ESC-50 dataset](https://github.com/karolpiczak/ESC-50) contains 2,000 five-second clips across 50 balanced classes. SoundSight preserves its supplied test folds instead of randomly splitting clips. Although the dataset describes source-grouped folds, the pinned metadata contains four source IDs crossing folds under different classes (`131943`, `134049`, `209698`, `234879`). For each test fold, SoundSight excludes training clips with any matching source ID. This stricter protocol differs slightly from standard official-fold benchmarks; exclusions appear in the report.

## Run it

Use Python 3.11 or 3.12. From this repository's root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
soundsight download
soundsight train --cv
streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`; the remaining commands are identical. The dataset download is approximately 600 MB and training extracts features from all 2,000 recordings. Allow several minutes or longer depending on CPU. If you already have ESC-50, use `soundsight train --data /path/to/ESC-50 --cv` with `audio/` and `meta/esc50.csv` under that directory. Acquisition refuses to overwrite an existing destination.

For one quicker initial experiment, use `soundsight train` (fold 5 held out). Run `soundsight train --cv` for all five folds. Each run replaces outputs in `artifacts/`; use `--out experiments/run-name` to preserve separate experiments. The app reads `artifacts/model.joblib`.

If the archive server stalls, use `soundsight download --individual` to obtain the same pinned revision through individual official files. A failed download may leave a partial destination; rename that directory before retrying. The downloader never treats a partial directory as a complete dataset.

## How it works

1. Validate the full metadata: unique filenames, 50 classes, five folds and eight examples of every class in each fold. Purge any overlapping source IDs from each training split.
2. Decode MP3, WAV, FLAC, OGG or AIFF, average stereo channels, resample to 22,050 Hz, and pad or crop to five seconds. Reject empty, silent, invalid, or over-30-second audio. No peak normalization is applied. The preview plays the processed five-second segment. MP3/OGG are lossy, so their predictions can differ from an uncompressed original.
3. Extract 64-band log-mel energies, 20 MFCCs, MFCC deltas, spectral centroid/bandwidth/rolloff, zero-crossing rate, and RMS. Pool each feature's mean and standard deviation over time into a 218-value vector. The app uses exactly the same functions.
4. Fit a `StandardScaler` and RBF support-vector classifier (`C=10`, `gamma=scale`). The scaler learns only from the training folds. Hyperparameters are fixed for this baseline; there is no test-fold tuning or augmentation.
5. Evaluate accuracy, macro F1, per-class precision/recall/F1, and a 50×50 confusion matrix. Cross-validation trains five independent pipelines and evaluates each clip exactly once.

The demo artifact deliberately retains the model trained without fold 5 by default. It is not refitted on all data after evaluation. Top-five values are SVM decision scores, **not calibrated probabilities**. Different-class scores are useful for ranking but do not establish real-world confidence.

## Evaluation and reproducibility

Read `artifacts/metrics.json` for measured results from the actual run. If it does not exist, **no ESC-50 accuracy has been measured**. Synthetic tones in the unit tests validate software plumbing only and never count as environmental-sound performance.

The report records preprocessing, model parameters, package versions, Python version, metadata hash, an ordered audio-content hash, fold sizes, and per-fold metrics. Acquisition records its archive URL and SHA-256 in `data/ESC-50/provenance.json`. `requirements-lock.txt` pins the app environment, retaining the numerical package versions used for training. PyArrow is pinned to 24.0.0 for Community Cloud compatibility; the initial local run used 26.0.0, which the host replaces due to a known crash. Reproduce with the same Python version and `pip install -r requirements-lock.txt`, then `pip install -e . --no-deps`.

`mean_accuracy` is the arithmetic mean across evaluated folds; for a single-fold run it is only that fold's result. `std_accuracy` uses population standard deviation across folds, not a confidence interval. Do not compare a single-fold score directly with a paper's five-fold score. Future model selection should use nested validation within the training folds; leave the outer test fold untouched.

Generated outputs:

| File | Purpose |
| --- | --- |
| `artifacts/model.joblib` | Fitted scaler/classifier, label map, preprocessing config and fold provenance |
| `artifacts/metrics.json` | Measured metrics and reproducibility record |
| `artifacts/predictions.json` | Auditable filename-level held-out predictions |
| `artifacts/confusion_matrix.png` | Confusions for the demo model's held-out fold |

## Repository

```text
app.py                     Streamlit upload demo
src/soundsight/audio.py     Shared preprocessing and feature extraction
src/soundsight/dataset.py   Download and dataset checks
src/soundsight/model.py     Pipeline, fold splitting and inference
src/soundsight/cli.py       Reproducible training and evaluation
tests/                     Preprocessing, leakage and model round-trip checks
.github/workflows/         Automated tests without downloading ESC-50
artifacts/                 Model and evaluation outputs after training
data/                      Local dataset, excluded from Git
```

## Limitations and responsible presentation

ESC-50 is small and curated. Background noise, different microphones, overlapping events and unfamiliar sounds can hurt performance. The classifier always chooses among its known classes; it does not detect an unknown class or locate events in time. The demo analyzes only the first five seconds. It is an educational prototype, not a safety or accessibility alerting system.

Uploads are processed in memory; this app does not save them. Hosting providers may have their own infrastructure policies. Only load a model artifact you trained or trust: joblib uses pickle, which can execute code. Never expose arbitrary artifact upload/loading to visitors.

ESC-50's dataset license is **CC BY-NC 3.0**; ESC-10 has separate terms. Review the [upstream license and attribution information](https://github.com/karolpiczak/ESC-50#license) before redistributing data, publishing trained artifacts, or using commercially. This repository does not relicense the dataset. Cite Karol J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia 2015. Dataset metadata preserves per-recording attribution.

## Portfolio next steps

After inspecting per-class errors, compare this fixed baseline with a log-mel CNN or pretrained audio embeddings using the same outer folds. Record training cost and latency alongside accuracy. Add unknown-sound rejection only after designing and evaluating an appropriate separate test set. Include actual demo screenshots and measured results in a GitHub project description; do not invent an accuracy target as a result.

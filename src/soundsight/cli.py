import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .audio import AudioConfig, extract
from .dataset import download, records
from .model import make_model, split_fold


def train(root, out, cv=False, held_out=5):
    root, out = Path(root), Path(out)
    rows = records(root)
    out.mkdir(parents=True, exist_ok=True)
    config = AudioConfig()
    x = []
    manifest = hashlib.sha256()
    for i, row in enumerate(rows):
        path = root / "audio" / row["filename"]
        manifest.update(path.name.encode())
        manifest.update(hashlib.sha256(path.read_bytes()).digest())
        x.append(extract(path, config))
        if (i + 1) % 100 == 0:
            print(f"Features: {i + 1}/2000", flush=True)
    x = np.asarray(x)
    y = np.array([r["target"] for r in rows])
    folds = np.array([r["fold"] for r in rows])
    sources = np.array([r["src_file"] for r in rows])
    labels = {r["target"]: r["category"] for r in rows}
    reports, predictions = [], []
    for fold in (range(1, 6) if cv else [held_out]):
        tr, te = split_fold(folds, fold, sources)
        model = make_model().fit(x[tr], y[tr])
        pred = model.predict(x[te])
        excluded = np.flatnonzero((folds != fold) & np.isin(sources, sources[te]))
        report = {"fold": fold, "train_count": len(tr), "test_count": len(te),
                  "excluded_shared_source_files": [rows[i]["filename"] for i in excluded],
                  "accuracy": accuracy_score(y[te], pred),
                  "macro_f1": f1_score(y[te], pred, average="macro"),
                  "classification_report": classification_report(y[te], pred, labels=list(range(50)), target_names=[labels[i] for i in range(50)], output_dict=True, zero_division=0),
                  "confusion_matrix": confusion_matrix(y[te], pred, labels=list(range(50))).tolist()}
        reports.append(report)
        for idx, value in zip(te, pred):
            predictions.append({"filename": rows[idx]["filename"], "fold": fold, "actual": int(y[idx]), "predicted": int(value)})
        print(f"Fold {fold}: accuracy={report['accuracy']:.4f}, macro F1={report['macro_f1']:.4f}", flush=True)
        if fold == held_out:
            joblib.dump({"format_version": 1, "model": model, "audio_config": asdict(config), "labels": labels, "held_out_fold": fold, "training_folds": sorted(set(folds[tr].tolist()))}, out / "model.joblib")
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol": "official five-fold CV with shared-source training clips purged" if cv else "single official held-out fold with shared-source training clips purged",
              "audio_sha256": manifest.hexdigest(), "metadata_sha256": hashlib.sha256((root / "meta/esc50.csv").read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256((out / "model.joblib").read_bytes()).hexdigest(),
              "audio_config": asdict(config), "model": {"type": "StandardScaler + RBF SVC", "C": 10, "gamma": "scale", "break_ties": True},
              "environment": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "librosa", "soundfile"]}},
              "labels": labels, "folds": reports,
              "mean_accuracy": float(np.mean([r["accuracy"] for r in reports])),
              "std_accuracy": float(np.std([r["accuracy"] for r in reports])),
              "mean_macro_f1": float(np.mean([r["macro_f1"] for r in reports]))}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    # Save an inspectable matrix for the demo model's held-out fold.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = next(r for r in reports if r["fold"] == held_out)
    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(report["confusion_matrix"], cmap="Blues")
    ax.set(xticks=range(50), yticks=range(50), xticklabels=[labels[i] for i in range(50)], yticklabels=[labels[i] for i in range(50)], xlabel="Predicted", ylabel="Actual", title=f"SoundSight — held-out fold {held_out}")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=130)
    plt.close(fig)
    print(f"Saved model and evaluation to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="SoundSight reproducible baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("--destination", default="data/ESC-50")
    dl.add_argument("--individual", action="store_true", help="Download pinned individual files if archive delivery stalls")
    fit = sub.add_parser("train")
    fit.add_argument("--data", default="data/ESC-50")
    fit.add_argument("--out", default="artifacts")
    fit.add_argument("--cv", action="store_true", help="Evaluate each official fold once")
    fit.add_argument("--held-out", type=int, choices=range(1, 6), default=5)
    args = parser.parse_args()
    if args.command == "download":
        download(args.destination, args.individual)
    else:
        train(args.data, args.out, args.cv, args.held_out)


if __name__ == "__main__":
    main()

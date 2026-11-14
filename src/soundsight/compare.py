"""Compare fixed classifiers on the same features and source-isolated folds."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from .audio import extract
from .dataset import records
from .model import make_model, split_fold


def compare(x, y, folds, sources):
    reports = []
    for fold in range(1, 6):
        train, test = split_fold(folds, fold, sources)
        models = {'Dummy': DummyClassifier(strategy='most_frequent'),
                  'Linear SVM': make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=42, max_iter=20000)),
                  'RBF SVM': make_model()}
        for name, model in models.items():
            pred = model.fit(x[train], y[train]).predict(x[test])
            reports.append({'model': name, 'fold': fold, 'train_n': len(train), 'test_n': len(test),
                            'accuracy': accuracy_score(y[test], pred),
                            'macro_f1': f1_score(y[test], pred, average='macro', zero_division=0)})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/ESC-50')
    parser.add_argument('--out', default='reports')
    args = parser.parse_args()
    root, out = Path(args.data), Path(args.out)
    rows = records(root)
    x = np.asarray([extract(root / 'audio' / row['filename']) for row in rows])
    reports = compare(x, np.array([r['target'] for r in rows]), np.array([r['fold'] for r in rows]), np.array([r['src_file'] for r in rows]))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'comparison.json').write_text(json.dumps({'feature_sha256': hashlib.sha256(x.tobytes()).hexdigest(), 'results': reports}, indent=2))
    lines = ['# Fixed-classifier comparison', '', 'Identical 218 features and source-isolated folds. Scaling fits training rows only. Hyperparameters are fixed, not selected on these folds.', '', '| Model | Mean accuracy | Mean macro F1 |', '|---|---:|---:|']
    for name in ['Dummy', 'Linear SVM', 'RBF SVM']:
        group = [r for r in reports if r['model'] == name]
        lines.append(f"| {name} | {np.mean([r['accuracy'] for r in group]):.4f} | {np.mean([r['macro_f1'] for r in group]):.4f} |")
    lines += ['', 'This isolates classifier choice, not the contribution of individual feature blocks. Feature ablations would require another controlled experiment.']
    (out / 'COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

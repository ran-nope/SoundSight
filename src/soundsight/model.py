"""Fold-safe fitting and inference."""
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .audio import AudioConfig, extract, features


def make_model():
    # Fixed hyperparameters: do not tune against reported held-out folds.
    return make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", gamma="scale", decision_function_shape="ovr", break_ties=True))


def split_fold(folds, held_out, sources=None):
    if held_out not in range(1, 6):
        raise ValueError("Fold must be 1..5.")
    test = np.asarray(folds) == held_out
    if not test.any() or test.all():
        raise ValueError("Both training and held-out examples are required.")
    train = ~test
    if sources is not None:
        sources = np.asarray(sources)
        if sources.shape != test.shape:
            raise ValueError("Source IDs must match the fold array.")
        train &= ~np.isin(sources, sources[test])
    if not train.any():
        raise ValueError("No training examples remain after source isolation.")
    return np.flatnonzero(train), np.flatnonzero(test)


def predict(source, bundle, top_k=5):
    config = AudioConfig(**bundle["audio_config"])
    x = extract(source, config)[None, :]
    return rank_features(x, bundle, top_k)


def predict_audio(y, bundle, top_k=5):
    """Predict from the already preprocessed waveform used by the plots."""
    config = AudioConfig(**bundle["audio_config"])
    return rank_features(features(y, config)[None, :], bundle, top_k)


def rank_features(x, bundle, top_k):
    scores = bundle["model"].decision_function(x)[0]
    order = np.argsort(scores)[::-1][:top_k]
    return [{"sound": bundle["labels"][int(bundle["model"].classes_[i])],
             "score": float(scores[i])} for i in order]


def load_bundle(path):
    # joblib/pickle can execute code. Only load artifacts you trained or trust.
    bundle = joblib.load(path)
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported model artifact. Retrain with this version.")
    return bundle

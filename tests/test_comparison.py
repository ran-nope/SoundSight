import numpy as np
from soundsight.compare import compare


def test_comparisons_share_fold_sizes_and_reproduce():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(50, 218))
    y = np.tile([0, 1], 25)
    folds = np.repeat(np.arange(1, 6), 10)
    sources = np.arange(50)
    results = compare(x, y, folds, sources)
    assert len(results) == 15
    assert results == compare(x, y, folds, sources)
    assert {(row['train_n'], row['test_n']) for row in results} == {(40, 10)}
    assert {row['model'] for row in results} == {'Dummy', 'Linear SVM', 'RBF SVM'}

# Fixed-classifier comparison

Identical 218 features and source-isolated folds. Scaling fits training rows only. Hyperparameters are fixed, not selected on these folds.

| Model | Mean accuracy | Mean macro F1 |
|---|---:|---:|
| Dummy | 0.0200 | 0.0008 |
| Linear SVM | 0.4620 | 0.4497 |
| RBF SVM | 0.5755 | 0.5626 |

This isolates classifier choice, not the contribution of individual feature blocks. Feature ablations would require another controlled experiment.

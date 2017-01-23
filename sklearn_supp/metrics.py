import sklearn.metrics
import numpy as np


def brier_score_loss_several(y_true, y_prob, sample_weight=None):
    """Compute the brier score for each label.

    y_true: array, shape (n_samples,)

    y_prob: array, shape (n_samples, n_labels)

    sample_weight: array-like, shape (n_samples,) or None, optional

    Returns:
    scores: float array, shape (n_labels,)
        contains the Brier score for each label.
    """
    n = y_prob.shape[1]
    result = [
        sklearn.metrics.brier_score_loss(
            y_true, y_prob[:, i], sample_weight=sample_weight, pos_label=i)
        for i in range(n)
    ]
    return np.array(result)


def brier_score_loss_total(y_true, y_prob, sample_weight=None):
    """Compute the total Brier score, i.e., the mean of the Brier scores for
    each label."""
    return np.mean(
        brier_score_loss_several(
            y_true, y_prob, sample_weight=None))

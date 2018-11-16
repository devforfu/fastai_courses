def mapk(y_pred, y_true, k):
    """Precision metric which compares several guesses with a single ground-truth value.

    Adapted from:
        https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

    """
    top_k = y_pred.argsort(descending=True)[:, :k]
    matched = top_k == y_true.view(-1, 1)
    map_k = 1 / (matched.argmax(dim=1) + 1).float()
    return map_k.mean()


def map3(y_pred, y_true):
    return mapk(y_pred, y_true, k=3)

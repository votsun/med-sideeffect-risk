from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)

def evaluate(model, X, y, threshold: float = 0.5):
    """
    Evaluate a probabilistic classifier with AUROC, AUPRC and accuracy.

    Parameters:
    model : fitted sklearn-like model with predict_proba
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    threshold : float, optional
        Probability threshold for converting probs to class labels.

    Returns:
    metrics : dict
        Keys: "AUROC", "AUPRC", "Accuracy".
    preds : ndarray
        Hard predictions (0 or 1).
    probs : ndarray
        Predicted probabilities for the positive class.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "AUROC": roc_auc_score(y, probs),
        "AUPRC": average_precision_score(y, probs),
        "Accuracy": accuracy_score(y, preds),
    }
    return metrics, preds, probs
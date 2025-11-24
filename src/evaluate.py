from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(model, X, y):
    preds = model.predict_proba(X)[:,1]
    return {
        "AUROC": roc_auc_score(y, preds),
        "AUPRC": average_precision_score(y, preds)
    }

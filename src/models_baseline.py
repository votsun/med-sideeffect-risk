from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_logreg(X_train, y_train):
    """
    L2-regularized logistic regression baseline.

    Returns a Pipeline that scales features and fits LogisticRegression.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=0,
        )),
    ])
    model.fit(X_train, y_train)
    return model

def train_logreg_l1(X_train, y_train):
    """
    L1-regularized logistic regression (sparse baseline).
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1",
            C=0.5,
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=0,
        )),
    ])
    model.fit(X_train, y_train)
    return model
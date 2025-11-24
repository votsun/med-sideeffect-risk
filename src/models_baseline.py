from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_logreg(X, y):
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X, y)
    return model

def train_rf(X, y):
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X, y)
    return model

def train_xgb(X, y):
    model = XGBClassifier(tree_method="hist", eval_metric="logloss")
    model.fit(X, y)
    return model

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import joblib

def train_mulitiouput_logistic_regression(X, y, save_path):
    lr_clf = LogisticRegression(solver="saga")
    mo_lr_clf = MultiOutputClassifier(lr_clf, n_jobs=-1).fit(X, y)

    joblib.dumb(mo_lr_clf, save_path)

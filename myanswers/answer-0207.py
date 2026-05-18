import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def class_accuracy(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    clases = np.unique(y)
    accuracies = []
    for c in clases:
        idx = (y == c)
        accuracies.append(np.sum(y_pred[idx] == y[idx]) / np.sum(idx))
    return np.array(accuracies)

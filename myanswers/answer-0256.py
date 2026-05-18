import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def calcular_pr_auc_cv(df: pd.DataFrame, target_col: str, cv_folds: int = 5) -> float:
    """
    Calcula el promedio de PR-AUC usando validación cruzada.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataframe con features y columna target
    target_col : str
        Nombre de la columna target
    cv_folds : int, default=5
        Número de folds para validación cruzada

    Retorna
    -------
    float
        PR-AUC promedio en los cv_folds
    """
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    X_scaled = StandardScaler().fit_transform(X)
    modelo = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
    )
    scores = cross_val_score(modelo, X_scaled, y, cv=cv_folds, scoring='average_precision')
    return float(scores.mean())

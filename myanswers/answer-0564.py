import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def detectar_correlacion_inestable(
    df=None,
    fecha_col=None,
    target_col=None,
    umbral_diff=0.3,
    **kwargs,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["feature", "corr_s1", "corr_s2", "diff_abs", "es_inestable"])

    if df is None or fecha_col is None or target_col is None:
        return empty

    df = df.copy()
    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df["_semestre"] = df[fecha_col].dt.to_period("6M")

    semestres = sorted(df["_semestre"].unique())
    if len(semestres) < 2:
        return empty

    s1, s2 = semestres[0], semestres[1]

    feature_cols = [
        c for c in df.columns
        if c not in (fecha_col, target_col, "_semestre")
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    def correlaciones(sub):
        X = sub[feature_cols].values
        y = sub[target_col].values
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_imp = imputer.fit_transform(X)
        corrs = []
        for j in range(X_imp.shape[1]):
            cc = np.corrcoef(X_imp[:, j], y)[0, 1]
            corrs.append(cc)
        return corrs

    corrs_s1 = correlaciones(df[df["_semestre"] == s1])
    corrs_s2 = correlaciones(df[df["_semestre"] == s2])

    result = pd.DataFrame({
        "feature": feature_cols,
        "corr_s1": corrs_s1,
        "corr_s2": corrs_s2,
    })
    result["diff_abs"] = (result["corr_s1"] - result["corr_s2"]).abs()
    result["es_inestable"] = result["diff_abs"] > umbral_diff
    result = result.sort_values("diff_abs", ascending=False).reset_index(drop=True)
    return result

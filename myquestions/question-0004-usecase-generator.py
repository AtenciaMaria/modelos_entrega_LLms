import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import sys
sys.stdout.reconfigure(encoding='utf-8')



# ─────────────────────────────────────────────────────────────────────────────
# Generador de casos de uso — independiente de calcular_pr_auc_cv
# ─────────────────────────────────────────────────────────────────────────────

def generar_caso_de_uso_calcular_pr_auc_cv():
    """
    Genera un par (input, output) aleatorio para la función
    'calcular_pr_auc_cv'.

    No recibe argumentos y NO depende de 'calcular_pr_auc_cv':
    el output se calcula directamente con sklearn, numpy y pandas.

    Retorna
    -------
    input  : dict   con claves 'df' (pd.DataFrame), 'target_col' (str)
                    y 'cv_folds' (int)
    output : float  PR-AUC promedio esperado para ese input
    """
    rng = np.random.default_rng()          # semilla distinta en cada llamada

    # ── 1. Parámetros del dataset ─────────────────────────────────────────────
    n_samples    = int(rng.integers(200, 701))    # entre 200 y 700 filas
    n_features   = int(rng.integers(3, 11))       # entre 3 y 10 features
    n_informative = int(rng.integers(2, n_features + 1))
    n_redundant  = int(rng.integers(0, max(1, (n_features - n_informative) + 1)))

    # Desbalance de clases: clase minoritaria entre 5% y 35%
    minority_frac = float(rng.uniform(0.05, 0.35))
    weights       = [round(1 - minority_frac, 2), round(minority_frac, 2)]

    dataset_seed  = int(rng.integers(0, 10_000))

    X_raw, y_raw = make_classification(
        n_samples     = n_samples,
        n_features    = n_features,
        n_informative = n_informative,
        n_redundant   = n_redundant,
        weights       = weights,
        flip_y        = 0.01,          # mínimo ruido de etiquetas
        random_state  = dataset_seed,
    )

    # ── 2. Número de folds ────────────────────────────────────────────────────
    cv_folds = int(rng.choice([3, 5, 7, 10]))

    # ── 3. Construir DataFrame ────────────────────────────────────────────────
    target_col  = "target"
    col_names   = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X_raw, columns=col_names)
    df[target_col] = y_raw

    # ── 4. Construir input ────────────────────────────────────────────────────
    caso_input = {
        "df"         : df,
        "target_col" : target_col,
        "cv_folds"   : cv_folds,
    }

    # ── 5. Calcular output SIN llamar a calcular_pr_auc_cv ────────────────────
    # 5a. Separar X e y
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # 5b. Escalar
    X_scaled = StandardScaler().fit_transform(X)

    # 5c. Modelo
    modelo = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
    )

    # 5d. Validación cruzada
    scores = cross_val_score(
        modelo, X_scaled, y,
        cv=cv_folds,
        scoring='average_precision',
    )

    # 5e. Promedio
    caso_output = float(scores.mean())

    return caso_input, caso_output


# ─────────────────────────────────────────────────────────────────────────────
# Demo: generar y mostrar 4 casos distintos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i in range(1, 5):
        inp, out = generar_caso_de_uso_calcular_pr_auc_cv()

        clase_counts = inp['df'][inp['target_col']].value_counts().to_dict()
        minority_pct = round(100 * min(clase_counts.values()) / sum(clase_counts.values()), 1)

        print(f"─── Caso {i} ───────────────────────────────────────────")
        print(f"  df.shape         : {inp['df'].shape}")
        print(f"  desbalance       : clase minoritaria = {minority_pct}%  {clase_counts}")
        print(f"  cv_folds         : {inp['cv_folds']}")
        print(f"  OUTPUT (PR-AUC)  : {out:.6f}  (tipo: {type(out).__name__})")
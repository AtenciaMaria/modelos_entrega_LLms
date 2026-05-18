import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Generador de casos de uso
# ─────────────────────────────────────────────────────────────────────────────

def generar_caso_de_uso_entrenar_evaluar_ridge():
    """
    Genera un par (input, output) aleatorio para la función
    'entrenar_evaluar_ridge'.

    No recibe argumentos y NO depende de 'entrenar_evaluar_ridge':
    el output se calcula directamente con sklearn, numpy y pandas.

    Retorna
    -------
    input  : dict   con claves 'df' (pd.DataFrame) y 'target_col' (str)
    output : float  MAE esperado para ese input
    """
    rng = np.random.default_rng()          # semilla distinta en cada llamada

    # ── 1. Parámetros del dataset ─────────────────────────────────────────────
    n_samples    = int(rng.integers(100, 601))   # entre 100 y 600 filas
    n_features   = int(rng.integers(3, 11))      # entre 3 y 10 features
    n_informative = int(rng.integers(2, n_features + 1))
    noise        = float(rng.uniform(5.0, 50.0)) # nivel de ruido del target
    dataset_seed = int(rng.integers(0, 10_000))

    X_raw, y_raw = make_regression(
        n_samples     = n_samples,
        n_features    = n_features,
        n_informative = n_informative,
        noise         = noise,
        random_state  = dataset_seed,
    )

    # ── 2. Construir DataFrame con nombres de columna aleatorios ──────────────
    feature_names = [f"feature_{i}" for i in range(n_features)]
    target_col    = "precio_vivienda"

    df = pd.DataFrame(X_raw, columns=feature_names)
    df[target_col] = y_raw

    # ── 3. Construir input ────────────────────────────────────────────────────
    caso_input = {
        "df"         : df,
        "target_col" : target_col,
    }

    # ── 4. Calcular output SIN llamar a entrenar_evaluar_ridge ────────────────
    # 4a. Separar X e y
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # 4b. Split 80/20  (mismo random_state=42 que exige el enunciado)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4c. Escalar (fit solo en train)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 4d. Entrenar Ridge
    modelo = Ridge(alpha=1.0)
    modelo.fit(X_train, y_train)

    # 4e. MAE en test
    y_pred      = modelo.predict(X_test)
    caso_output = float(mean_absolute_error(y_test, y_pred))

    return caso_input, caso_output


# ─────────────────────────────────────────────────────────────────────────────
# Demo: generar y mostrar 3 casos distintos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i in range(1, 4):
        inp, out = generar_caso_de_uso_entrenar_evaluar_ridge()
        print(f"─── Caso {i} ───────────────────────────────────────────")
        print(f"  INPUTS")
        print(f"  df.shape     : {inp['df'].shape}")
        print(f"  target_col   : '{inp['target_col']}'")
        print(f"  columnas     : {list(inp['df'].columns)}")
        print(f"  OUTPUT (MAE) : {out:.6f}  (tipo: {type(out).__name__})")
        print()

import numpy as np 
from sklearn.datasets import make_classification  
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import f1_score  
import sys
sys.stdout.reconfigure(encoding='utf-8')


# ─────────────────────────────────────────────────────────────────────────────
# Generador de casos de uso aleatorios
# ─────────────────────────────────────────────────────────────────────────────

def generar_caso_de_uso_encontrar_umbral():
    """
    Genera un par (input, output) aleatorio para la función
    'encontrar_umbral_optimo'.

    No recibe argumentos y NO depende de 'encontrar_umbral_optimo':
    el output se calcula directamente con sklearn y numpy.

    Retorna
    -------
    input  : dict   con claves X, y, n_umbrales, random_state
    output : float  umbral óptimo esperado para ese input
    """
    rng = np.random.default_rng()          # semilla distinta en cada llamada

    # ── 1. Generar dataset aleatorio ──────────────────────────────────────────
    n_samples     = int(rng.integers(200, 801))
    n_features    = int(rng.integers(4, 13))
    n_informative = int(rng.integers(2, max(3, n_features - 1)))
    max_redundant = min(n_informative // 2, n_features - n_informative)
    n_redundant   = int(rng.integers(0, max(1, max_redundant + 1)))
    weights_pos   = float(rng.uniform(0.30, 0.70))
    weights       = [round(weights_pos, 2), round(1 - weights_pos, 2)]
    dataset_seed  = int(rng.integers(0, 10_000))

    X, y = make_classification(
        n_samples     = n_samples,
        n_features    = n_features,
        n_informative = n_informative,
        n_redundant   = n_redundant,
        weights       = weights,
        random_state  = dataset_seed,
    )

    # ── 2. Elegir parámetros de la función ────────────────────────────────────
    n_umbrales   = int(rng.choice([20, 30, 50, 75, 100]))
    random_state = int(rng.integers(0, 200))

    # ── 3. Construir input ────────────────────────────────────────────────────
    caso_input = {
        "X"            : X,
        "y"            : y,
        "n_umbrales"   : n_umbrales,
        "random_state" : random_state,
    }

    # ── 4. Calcular output SIN llamar a encontrar_umbral_optimo ───────────────
    # 4a. Escalar
    X_scaled = StandardScaler().fit_transform(X)

    # 4b. Entrenar
    modelo = LogisticRegression(max_iter=1000, random_state=random_state)
    modelo.fit(X_scaled, y)

    # 4c. Probabilidades de la clase positiva
    probs = modelo.predict_proba(X_scaled)[:, 1]

    # 4d. Barrer umbrales y quedarse con el de mayor F1
    umbrales     = np.linspace(0.01, 0.99, n_umbrales)
    mejor_f1     = -1.0
    mejor_umbral = umbrales[0]

    for u in umbrales:
        y_pred = (probs >= u).astype(int)
        f1 = f1_score(y, y_pred, zero_division=0)
        if f1 > mejor_f1:
            mejor_f1     = f1
            mejor_umbral = u

    caso_output = round(float(mejor_umbral), 4)

    return caso_input, caso_output


# ─────────────────────────────────────────────────────────────────────────────
# Demo: generar y mostrar 3 casos distintos
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for i in range(1, 4):
        inp, out = generar_caso_de_uso_encontrar_umbral()
        print(f"─── Caso {i} ───────────────────────────────────────────")
        print(f"  INPUTS")
        print(f"  X.shape      : {inp['X'].shape}")
        print(f"  y único vals : {np.unique(inp['y'], return_counts=True)}")
        print(f"  n_umbrales   : {inp['n_umbrales']}")
        print(f"  random_state : {inp['random_state']}")
        print(f"  OUTPUT       : {out}  (tipo: {type(out).__name__})")
        print()
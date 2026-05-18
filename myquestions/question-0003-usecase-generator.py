import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Generador de casos de uso
# ─────────────────────────────────────────────────────────────────────────────

def generar_caso_de_uso_calcular_numero_condicion():
    """
    Genera un par (input, output) aleatorio para la función
    'calcular_numero_condicion'.

    No recibe argumentos y NO depende de 'calcular_numero_condicion':
    el output se calcula directamente con numpy y pandas.

    Retorna
    -------
    input  : dict   con clave 'df' (pd.DataFrame con columnas numéricas)
    output : float  número de condición esperado para ese input
    """
    rng = np.random.default_rng()          # semilla distinta en cada llamada

    # ── 1. Dimensiones del dataset ────────────────────────────────────────────
    n_samples  = int(rng.integers(80, 401))   # entre 80 y 400 filas
    n_features = int(rng.integers(2, 9))      # entre 2 y 8 columnas numéricas

    # ── 2. Elegir régimen de multicolinealidad ────────────────────────────────
    # 0 → sin correlación, 1 → baja, 2 → moderada, 3 → alta
    regimen = int(rng.integers(0, 4))

    dataset_seed = int(rng.integers(0, 10_000))
    rs = np.random.RandomState(dataset_seed)

    if regimen == 0:
        # Variables independientes: cada columna es ruido puro
        data = rs.randn(n_samples, n_features)

    elif regimen == 1:
        # Correlación baja: base común débil + ruido dominante
        base = rs.randn(n_samples, 1)
        data = 0.2 * base + 0.8 * rs.randn(n_samples, n_features)

    elif regimen == 2:
        # Correlación moderada: base común equilibrada con ruido
        base = rs.randn(n_samples, 1)
        data = 0.6 * base + 0.4 * rs.randn(n_samples, n_features)

    else:
        # Correlación alta: base casi perfecta + ruido mínimo
        base = rs.randn(n_samples, 1)
        data = 0.95 * base + 0.05 * rs.randn(n_samples, n_features)

    # ── 3. Construir DataFrame con nombres de columna descriptivos ─────────────
    col_names = [f"var_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=col_names)

    # ── 4. Construir input ────────────────────────────────────────────────────
    caso_input = {"df": df}

    # ── 5. Calcular output SIN llamar a calcular_numero_condicion ─────────────
    # 5a. Columnas numéricas
    numericas = df.select_dtypes(include=[np.number])

    # 5b. Matriz de correlación
    corr_matrix = numericas.corr()

    # 5c. Autovalores
    eigenvalues = np.linalg.eigvals(corr_matrix.values)

    # 5d. Número de condición
    abs_eigs = np.abs(eigenvalues)
    eig_max  = abs_eigs.max()
    eig_min  = abs_eigs.min()

    if eig_min == 0:
        caso_output = float(np.inf)
    else:
        caso_output = float(np.sqrt(eig_max / eig_min))

    return caso_input, caso_output


# ─────────────────────────────────────────────────────────────────────────────
# Demo: generar y mostrar 4 casos distintos (uno por régimen)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    regimenes = ["sin correlación", "baja", "moderada", "alta"]

    for i in range(1, 5):
        inp, out = generar_caso_de_uso_calcular_numero_condicion()
        print(f"─── Caso {i} ───────────────────────────────────────────")
        print(f"  INPUTS")
        print(f"  df.shape              : {inp['df'].shape}")
        print(f"  columnas              : {list(inp['df'].columns)}")
        print(f"  OUTPUT (nº condición) : {out:.6f}  (tipo: {type(out).__name__})")

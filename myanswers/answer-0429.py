import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def detectar_vibracion_anomala(
    df: pd.DataFrame,
    n_components: int,
    umbral_percentil: float = 95,
) -> np.ndarray:
    """
    Detecta vibraciones anómalas en un DataFrame usando PCA.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con columnas numéricas (y potencialmente categóricas).
    n_components : int
        Número de componentes principales a usar en PCA.
    umbral_percentil : float, default 95
        Percentil para definir el umbral de anomalía.

    Retorna:
    --------
    np.ndarray
        Array booleano donde True indica una anomalía (fila anómala).
    """
    # Seleccionar solo columnas numéricas
    X = df.select_dtypes(include=[np.number]).values

    # Normalizar los datos
    X_scaled = StandardScaler().fit_transform(X)

    # Aplicar PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_reduced)

    # Calcular error cuadrático medio por fila
    errores = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

    # Calcular umbral y detectar anomalías
    umbral = np.percentile(errores, umbral_percentil)
    return errores > umbral

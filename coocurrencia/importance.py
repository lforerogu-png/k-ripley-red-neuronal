"""Importancia de variables en la red neuronal.

Dos estrategias complementarias, comparables con los coeficientes de la
regresión logística:

* **Permutation feature importance**: se permuta cada variable en el conjunto
  de prueba y se mide la caída de exactitud.
* **Análisis de sensibilidad**: se barre cada variable en su rango fijando las
  demás en su valor medio, midiendo cómo cambia la clase/probabilidad predicha.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .dataset import FEATURES


def permutation_importance(
    resultado_entrenamiento,
    test: pd.DataFrame,
    target: str = "clase_celda",
    features: Sequence[str] = FEATURES,
    n_repeticiones: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Importancia por permutación sobre el conjunto de prueba.

    Parameters
    ----------
    resultado_entrenamiento:
        Objeto :class:`~coocurrencia.models.ResultadoEntrenamiento`.
    """
    rng = np.random.default_rng(seed)
    y = test[target].to_numpy()
    base = accuracy_score(y, resultado_entrenamiento.predecir(test, features))

    filas = []
    for f in features:
        caidas = []
        for _ in range(n_repeticiones):
            df_perm = test.copy()
            df_perm[f] = rng.permutation(df_perm[f].to_numpy())
            acc = accuracy_score(
                y, resultado_entrenamiento.predecir(df_perm, features)
            )
            caidas.append(base - acc)
        filas.append({
            "variable": f,
            "importancia_media": float(np.mean(caidas)),
            "importancia_std": float(np.std(caidas)),
        })
    tabla = pd.DataFrame(filas).sort_values(
        "importancia_media", ascending=False
    ).reset_index(drop=True)
    tabla.attrs["accuracy_base"] = base
    return tabla


def analisis_sensibilidad(
    resultado_entrenamiento,
    test: pd.DataFrame,
    features: Sequence[str] = FEATURES,
    n_puntos: int = 25,
) -> dict:
    """Análisis de sensibilidad: barrido de cada variable con las demás fijas.

    Returns
    -------
    dict
        Mapea cada variable a un DataFrame con la rejilla de valores y la
        probabilidad media predicha de cada clase.
    """
    X = test[list(features)].to_numpy(dtype=float)
    medios = X.mean(axis=0)
    clases = resultado_entrenamiento.clases
    salida = {}

    for j, f in enumerate(features):
        lo, hi = X[:, j].min(), X[:, j].max()
        rejilla = np.linspace(lo, hi, n_puntos)
        base = np.tile(medios, (n_puntos, 1))
        base[:, j] = rejilla
        Xs = resultado_entrenamiento.scaler.transform(base)
        proba = resultado_entrenamiento.modelo.predict_proba(Xs)
        df = pd.DataFrame(proba, columns=[str(c) for c in clases])
        df.insert(0, f, rejilla)
        salida[f] = df
    return salida


def comparar_importancias(
    perm_mlp: pd.DataFrame, coefs_logistica: pd.DataFrame
) -> pd.DataFrame:
    """Une el ranking de permutación (MLP) con |coef| de la logística."""
    a = perm_mlp[["variable", "importancia_media"]].copy()
    a["rank_mlp"] = a["importancia_media"].rank(ascending=False)
    b = coefs_logistica.copy()
    b["abs_coef"] = b["coef"].abs() if "coef" in b else b.get("abs_coef")
    b = b[["variable", "abs_coef"]]
    b["rank_logit"] = b["abs_coef"].rank(ascending=False)
    return a.merge(b, on="variable", how="outer")

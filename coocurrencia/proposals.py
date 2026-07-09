"""Propuestas propias (extensiones no pedidas explícitamente).

Contiene varias extensiones metodológicas para la sección de "propuesta
libre" del reporte:

1. Comparación con Random Forest y SVM usando las mismas particiones/métricas.
2. Análisis de robustez ante ruido en los valores de K.
3. Reducción de dimensión (PCA y t-SNE) de las entradas.
4. Regularización: efecto de ``early stopping`` y de la fuerza L2 (``alpha``)
   sobre el sobreajuste (alternativa a dropout, no disponible en sklearn).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataset import FEATURES
from .models import Particion, preparar_xy


def comparar_modelos_alternativos(
    particion: Particion,
    target: str = "clase_celda",
    features: Sequence[str] = FEATURES,
    seed: int = 42,
) -> pd.DataFrame:
    """Compara MLP, Random Forest y SVM sobre la misma partición."""
    Xtr, ytr = preparar_xy(particion.train, target, features)
    Xte, yte = preparar_xy(particion.test, target, features)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    modelos = {
        "MLP(16x8)": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=400,
                                    random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=300,
                                                random_state=seed),
        "SVM(rbf)": SVC(kernel="rbf", probability=False, random_state=seed),
    }
    filas = []
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    for nombre, m in modelos.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(Xtr_s, ytr)
        pred = m.predict(Xte_s)
        filas.append({
            "modelo": nombre,
            "accuracy": accuracy_score(yte, pred),
            "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(yte, pred, average="weighted", zero_division=0),
        })
    return pd.DataFrame(filas).sort_values("accuracy", ascending=False).reset_index(drop=True)


def robustez_ante_ruido(
    particion: Particion,
    target: str = "clase_celda",
    features: Sequence[str] = FEATURES,
    niveles_ruido: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
    columnas_k: Sequence[str] = ("k1_n", "k2_n", "k12_n"),
    seed: int = 42,
) -> pd.DataFrame:
    """Evalúa la caída de exactitud al añadir ruido gaussiano a las K.

    Entrena una vez sobre datos limpios y evalúa sobre un test perturbado con
    ruido de desviación proporcional al nivel indicado.
    """
    Xtr, ytr = preparar_xy(particion.train, target, features)
    scaler = StandardScaler().fit(Xtr)
    modelo = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=400,
                           random_state=seed)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        modelo.fit(scaler.transform(Xtr), ytr)

    rng = np.random.default_rng(seed)
    yte = particion.test[target].to_numpy()
    idx_k = [list(features).index(c) for c in columnas_k if c in features]
    filas = []
    for nivel in niveles_ruido:
        Xte = particion.test[list(features)].to_numpy(dtype=float).copy()
        for j in idx_k:
            sd = Xte[:, j].std() or 1.0
            Xte[:, j] += rng.normal(0, nivel * sd, size=len(Xte))
        acc = accuracy_score(yte, modelo.predict(scaler.transform(Xte)))
        filas.append({"nivel_ruido": nivel, "accuracy": acc})
    return pd.DataFrame(filas)


def reducir_dimension(
    datos_coloreados: pd.DataFrame,
    target: str = "clase_celda",
    features: Sequence[str] = FEATURES,
    metodo: str = "pca",
    seed: int = 42,
    max_muestras: int = 2000,
) -> pd.DataFrame:
    """Proyección 2D de las entradas con PCA o t-SNE.

    Returns
    -------
    pandas.DataFrame con columnas ``dim1``, ``dim2`` y ``clase``.
    """
    df = datos_coloreados
    if len(df) > max_muestras:
        df = df.sample(max_muestras, random_state=seed)
    X = StandardScaler().fit_transform(df[list(features)].to_numpy(dtype=float))
    if metodo == "tsne":
        red = TSNE(n_components=2, random_state=seed, init="pca",
                   perplexity=min(30, max(5, len(df) // 100)))
        Z = red.fit_transform(X)
    else:
        red = PCA(n_components=2, random_state=seed)
        Z = red.fit_transform(X)
    out = pd.DataFrame({"dim1": Z[:, 0], "dim2": Z[:, 1],
                        "clase": df[target].to_numpy()})
    if metodo == "pca":
        out.attrs["varianza_explicada"] = red.explained_variance_ratio_.tolist()
    return out


def efecto_regularizacion(
    particion: Particion,
    target: str = "clase_celda",
    features: Sequence[str] = FEATURES,
    alphas: Sequence[float] = (1e-5, 1e-3, 1e-1, 1.0),
    seed: int = 42,
) -> pd.DataFrame:
    """Compara early stopping y distintas fuerzas L2 (``alpha``).

    Alternativa a dropout (no soportado por sklearn): se contrasta el gap
    train-test para distintas intensidades de regularización y con/ sin
    parada temprana.
    """
    Xtr, ytr = preparar_xy(particion.train, target, features)
    Xte, yte = preparar_xy(particion.test, target, features)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    filas = []
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    for alpha in alphas:
        for early in (False, True):
            m = MLPClassifier(
                hidden_layer_sizes=(32, 16), alpha=alpha, max_iter=500,
                early_stopping=early, validation_fraction=0.15,
                n_iter_no_change=15, random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                m.fit(Xtr_s, ytr)
            acc_tr = accuracy_score(ytr, m.predict(Xtr_s))
            acc_te = accuracy_score(yte, m.predict(Xte_s))
            filas.append({
                "alpha": alpha,
                "early_stopping": early,
                "acc_train": acc_tr,
                "acc_test": acc_te,
                "gap": acc_tr - acc_te,
            })
    return pd.DataFrame(filas)

r"""Clasificación por simulaciones con índices de concordancia espacial.

Este módulo implementa un enfoque **por simulación** (complementario al enfoque
celda por celda de :mod:`coocurrencia.dataset`): cada simulación genera un par
de patrones (A, B) y se resume en cuatro índices de concordancia calculados
sobre la tabla de contingencia de colores de la grilla:

* **Jaccard**  = verde / (azul + amarillo + verde)
* **Dice**     = 2·verde / (2·verde + azul + amarillo)
* **MCC**      = coeficiente de correlación de Matthews (tabla 2×2)
* **Rand**     = (verde + blanco) / total

donde, para cada celda de la grilla:

======  ==================  =============================
símbolo  color               significado
======  ==================  =============================
``a``    verde               ambos patrones presentes
``b``    azul                solo A
``c``    amarillo            solo B
``d``    blanco              ninguno (celda vacía)
======  ==================  =============================

Jaccard y Dice **excluyen** las celdas blancas (no dependen de ``d``). MCC y
Rand usan la tabla 2×2 completa (incluyen ``d``), como corresponde a su
definición estándar: miden concordancia sobre todo el espacio de celdas.

La variable de salida binaria proviene de la **clasificación de la K de Ripley
cruzada**: concordancia (1) o no concordancia (0), con dos criterios
seleccionables (ver :func:`etiqueta_concordancia`).

Adicionalmente se ofrece el estadístico **Join Count** (recuento de uniones
BB) sobre la grilla binaria de ocupación como quinta variable candidata.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import ripley
from .patterns import generar_patron
from .viz import matriz_estados

# Nombres de las cuatro variables base y la quinta opcional.
VARIABLES_BASE = ["jaccard", "dice", "mcc", "rand"]
VARIABLE_JOIN = "join_count"

# Definiciones de concordancia para la etiqueta binaria.
CONCORDANCIA_SOLO_VERDE = "solo_verde"
CONCORDANCIA_CUALQUIERA = "verde_azul_amarillo"


def tabla_contingencia_colores(p1, p2, n_grid: int = 30) -> dict:
    """Cuenta celdas por color: ``a`` verde, ``b`` azul, ``c`` amarillo, ``d`` blanco."""
    estado = matriz_estados(p1, p2, n_grid)
    a = int(np.sum(estado == 3))   # ambos (verde)
    b = int(np.sum(estado == 1))   # solo A (azul)
    c = int(np.sum(estado == 2))   # solo B (amarillo)
    d = int(np.sum(estado == 0))   # ninguno (blanco)
    return {"verde": a, "azul": b, "amarillo": c, "blanco": d}


def indices_concordancia(p1, p2, n_grid: int = 30) -> dict:
    """Calcula los cuatro índices de concordancia sobre la grilla."""
    cont = tabla_contingencia_colores(p1, p2, n_grid)
    a, b, c, d = cont["verde"], cont["azul"], cont["amarillo"], cont["blanco"]

    jaccard = a / (a + b + c) if (a + b + c) > 0 else 0.0
    dice = 2 * a / (2 * a + b + c) if (2 * a + b + c) > 0 else 0.0

    # MCC estándar de tabla 2×2 (incluye celdas blancas d).
    denom_mcc = float((a + b) * (a + c) * (d + b) * (d + c))
    mcc = (a * d - b * c) / np.sqrt(denom_mcc) if denom_mcc > 0 else 0.0

    total = a + b + c + d
    rand = (a + d) / total if total > 0 else 0.0

    return {
        "jaccard": float(jaccard),
        "dice": float(dice),
        "mcc": float(mcc),
        "rand": float(rand),
        **cont,
    }


def join_count_bb(p1, p2, n_grid: int = 30) -> float:
    """Estadístico Join Count (uniones BB) sobre la grilla binaria de ocupación.

    Se define la grilla binaria: celda "ocupada" si tiene al menos un patrón
    (A o B), "vacía" en otro caso. El recuento BB cuenta las aristas (vecindad
    rook: arriba/abajo/izquierda/derecha) que conectan dos celdas ocupadas.

    Se devuelve la **razón** entre las uniones BB observadas y las esperadas
    bajo ocupación aleatoria:

    .. math::
        \\text{ratio} = \\frac{BB_{obs}}{J \\cdot n_1 (n_1 - 1) / (n (n - 1))}

    donde ``J`` es el número total de aristas de la grilla, ``n`` el número de
    celdas y ``n_1`` las celdas ocupadas. Valores > 1 indican agregación
    espacial de la ocupación; ~1 aleatoriedad; < 1 dispersión.
    """
    estado = matriz_estados(p1, p2, n_grid)
    occ = estado > 0

    bb_horiz = int(np.sum(occ[:, :-1] & occ[:, 1:]))
    bb_vert = int(np.sum(occ[:-1, :] & occ[1:, :]))
    observado = bb_horiz + bb_vert

    n = n_grid * n_grid
    n1 = int(np.sum(occ))
    j_total = 2 * n_grid * (n_grid - 1)  # aristas rook en grilla NxN
    if n <= 1 or n1 < 2:
        return 0.0
    esperado = j_total * n1 * (n1 - 1) / (n * (n - 1))
    if esperado <= 0:
        return 0.0
    return float(observado / esperado)


def etiqueta_concordancia(
    clasificacion_k: str,
    definicion: str = CONCORDANCIA_SOLO_VERDE,
) -> int:
    """Convierte la clasificación de la K cruzada en etiqueta binaria.

    Parameters
    ----------
    clasificacion_k:
        ``"atraccion"``, ``"repulsion"`` o ``"sin_coocurrencia"``.
    definicion:
        * ``CONCORDANCIA_SOLO_VERDE``: concordancia (1) solo si hay atracción
          (coocurrencia positiva, celdas verdes dominantes).
        * ``CONCORDANCIA_CUALQUIERA``: concordancia (1) si hay cualquier
          asociación espacial no aleatoria (atracción o repulsión).
    """
    if definicion == CONCORDANCIA_CUALQUIERA:
        return int(clasificacion_k in ("atraccion", "repulsion"))
    return int(clasificacion_k == "atraccion")


def simular_dataset_concordancia(
    n_sim: int,
    tipo_a: str,
    tipo_b: str,
    n_pts_a: int,
    n_pts_b: int,
    n_grid: int = 30,
    r_ref: float = 0.2,
    tol: float = 0.15,
    definicion_concordancia: str = CONCORDANCIA_SOLO_VERDE,
    incluir_join_count: bool = True,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Genera un dataset con una fila por simulación.

    Cada simulación genera A y B de forma independiente con el tipo fijo
    elegido y calcula los índices de concordancia y la etiqueta binaria de la
    K de Ripley cruzada.

    Returns
    -------
    pandas.DataFrame
        Columnas: ``jaccard``, ``dice``, ``mcc``, ``rand``
        (+ ``join_count`` si procede), ``concordancia`` (etiqueta 0/1),
        y ``k12_clasificacion`` (texto).
    """
    rng = np.random.default_rng(seed)
    semillas = rng.integers(0, 2**31 - 1, size=(n_sim, 2))
    filas = []
    for i in range(n_sim):
        seed_a = int(semillas[i, 0])
        seed_b = int(semillas[i, 1])
        p1 = generar_patron(tipo_a, n_pts_a, seed_a, n_grid=n_grid)
        p2 = generar_patron(tipo_b, n_pts_b, seed_b, n_grid=n_grid)

        idx = indices_concordancia(p1, p2, n_grid)
        kc = ripley.k_cruzada(p1[["x", "y"]].to_numpy(), p2[["x", "y"]].to_numpy())
        cls_k = ripley.clasificar_bivariada_umbral(kc, r_ref=r_ref, tol=tol)
        etiqueta = etiqueta_concordancia(cls_k, definicion_concordancia)

        fila = {
            "sim_id": i,
            "jaccard": idx["jaccard"],
            "dice": idx["dice"],
            "mcc": idx["mcc"],
            "rand": idx["rand"],
        }
        if incluir_join_count:
            fila["join_count"] = join_count_bb(p1, p2, n_grid)
        fila["concordancia"] = etiqueta
        fila["k12_clasificacion"] = cls_k
        filas.append(fila)

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Entrenamiento con Keras (fallback a scikit-learn si TensorFlow no está)
# ---------------------------------------------------------------------------
def _entrenar_keras(X_tr, y_tr, X_te, epochs: int, seed: int):
    """Entrena una red densa con Keras. Devuelve (y_pred, y_proba, backend)."""
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    tf.random.set_seed(seed)
    n_features = X_tr.shape[1]
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    modelo.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"],
    )
    modelo.fit(X_tr, y_tr, epochs=epochs, batch_size=16, verbose=0)
    proba = modelo.predict(X_te, verbose=0).ravel()
    y_pred = (proba >= 0.5).astype(int)
    return y_pred, proba, "keras"


def _entrenar_sklearn(X_tr, y_tr, X_te, seed: int):
    """Fallback con scikit-learn MLPClassifier."""
    from sklearn.neural_network import MLPClassifier

    modelo = MLPClassifier(
        hidden_layer_sizes=(16, 8), activation="relu",
        solver="adam", max_iter=500, random_state=seed,
    )
    modelo.fit(X_tr, y_tr)
    if len(modelo.classes_) < 2:
        y_pred = modelo.predict(X_te)
        proba = np.full(len(X_te), float(modelo.classes_[0]))
        return y_pred.astype(int), proba, "sklearn"
    proba = modelo.predict_proba(X_te)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    return y_pred, proba, "sklearn"


def entrenar_red_concordancia(
    dataset: pd.DataFrame,
    variables: list[str],
    target: str = "concordancia",
    pct_train: float = 0.8,
    epochs: int = 80,
    seed: int = 42,
) -> dict:
    """Entrena la red (Keras o fallback) sobre filas de simulaciones.

    La partición ``pct_train`` se hace sobre las **filas** (simulaciones).

    Returns
    -------
    dict
        ``accuracy``, ``y_true``, ``y_pred``, ``n_test``, ``backend``.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    df = dataset.dropna(subset=variables + [target]).reset_index(drop=True)
    n = len(df)
    if n < 10:
        raise ValueError("Se requieren al menos 10 simulaciones para entrenar.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * pct_train))
    n_train = max(1, min(n_train, n - 1))
    tr_idx, te_idx = idx[:n_train], idx[n_train:]

    X = df[variables].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=int)

    scaler = StandardScaler().fit(X[tr_idx])
    X_tr = scaler.transform(X[tr_idx])
    X_te = scaler.transform(X[te_idx])
    y_tr, y_te = y[tr_idx], y[te_idx]

    if len(np.unique(y_tr)) < 2:
        # Una sola clase en entrenamiento: predice esa clase constante.
        y_pred = np.full(len(te_idx), int(y_tr[0]))
        backend = "constante"
    else:
        try:
            y_pred, _, backend = _entrenar_keras(X_tr, y_tr, X_te, epochs, seed)
        except Exception:  # noqa: BLE001  (TF ausente o error de backend)
            y_pred, _, backend = _entrenar_sklearn(X_tr, y_tr, X_te, seed)

    acc = accuracy_score(y_te, y_pred)
    return {
        "accuracy": float(acc),
        "y_true": y_te,
        "y_pred": np.asarray(y_pred, dtype=int),
        "n_test": len(te_idx),
        "n_train": len(tr_idx),
        "backend": backend,
    }

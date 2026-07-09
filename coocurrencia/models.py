"""Perceptrón multicapa y comparación sistemática de configuraciones.

Incluye, con el nivel de rigor del taller de Verticillium:

* Partición por **simulación** (no por celda) para evitar fuga de información,
  ya que las variables de K son a nivel de patrón (constantes dentro de una
  simulación). Se implementan los esquemas 70/15/15 y 80/20.
* Entrenamiento con seguimiento de la **pérdida de entrenamiento y validación
  por época** (curvas de aprendizaje) mediante ``partial_fit``.
* Comparación de arquitecturas: nº de capas ocultas (1, 2, 3), neuronas por
  capa y tasa de aprendizaje (0.001, 0.01, 0.1).
* Comparación de funciones de pérdida: entropía cruzada categórica vs. MSE.
* Validación cruzada k-fold agrupada por simulación.

Todas las evaluaciones se hacen sobre celdas coloreadas (espacio condicional).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from .dataset import FEATURES, filtrar_coloreadas


# ---------------------------------------------------------------------------
# Particiones por simulación
# ---------------------------------------------------------------------------
@dataclass
class Particion:
    """Conjuntos de entrenamiento/validación/prueba (celdas coloreadas)."""

    train: pd.DataFrame
    val: Optional[pd.DataFrame]
    test: pd.DataFrame
    esquema: str


def particion_por_sim(
    datos: pd.DataFrame,
    fracciones: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Particion:
    """Divide el dataset por ``sim_id`` en train/val/test o train/test.

    Parameters
    ----------
    fracciones:
        ``(train, val, test)`` para 3 particiones o ``(train, test)`` para 2.
    """
    sim_ids = np.array(sorted(datos["sim_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(sim_ids)
    n = len(sim_ids)

    if len(fracciones) == 3:
        n_tr = int(round(fracciones[0] * n))
        n_va = int(round(fracciones[1] * n))
        tr = sim_ids[:n_tr]
        va = sim_ids[n_tr:n_tr + n_va]
        te = sim_ids[n_tr + n_va:]
        esquema = "/".join(f"{int(round(f*100))}" for f in fracciones)
        return Particion(
            train=filtrar_coloreadas(datos[datos["sim_id"].isin(tr)]),
            val=filtrar_coloreadas(datos[datos["sim_id"].isin(va)]),
            test=filtrar_coloreadas(datos[datos["sim_id"].isin(te)]),
            esquema=esquema,
        )
    else:
        n_tr = int(round(fracciones[0] * n))
        tr = sim_ids[:n_tr]
        te = sim_ids[n_tr:]
        esquema = "/".join(f"{int(round(f*100))}" for f in fracciones)
        return Particion(
            train=filtrar_coloreadas(datos[datos["sim_id"].isin(tr)]),
            val=None,
            test=filtrar_coloreadas(datos[datos["sim_id"].isin(te)]),
            esquema=esquema,
        )


# ---------------------------------------------------------------------------
# Preparación de matrices X, y
# ---------------------------------------------------------------------------
def preparar_xy(
    df: pd.DataFrame, target: str = "clase_celda", features: Sequence[str] = FEATURES
):
    """Extrae ``X`` (features) e ``y`` (target) de un DataFrame de celdas."""
    X = df[list(features)].to_numpy(dtype=float)
    y = df[target].to_numpy()
    return X, y


# ---------------------------------------------------------------------------
# Entrenamiento con curvas de aprendizaje (train/val loss por época)
# ---------------------------------------------------------------------------
@dataclass
class ResultadoEntrenamiento:
    """MLP entrenado con sus curvas de pérdida y scaler asociado."""

    modelo: MLPClassifier
    scaler: StandardScaler
    clases: np.ndarray
    loss_train: list = field(default_factory=list)
    loss_val: list = field(default_factory=list)
    acc_train: list = field(default_factory=list)
    acc_val: list = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def predecir(self, df: pd.DataFrame, features: Sequence[str] = FEATURES):
        X = self.scaler.transform(df[list(features)].to_numpy(dtype=float))
        return self.modelo.predict(X)

    def predecir_proba(self, df: pd.DataFrame, features: Sequence[str] = FEATURES):
        X = self.scaler.transform(df[list(features)].to_numpy(dtype=float))
        return self.modelo.predict_proba(X)


def entrenar_mlp(
    train: pd.DataFrame,
    val: Optional[pd.DataFrame] = None,
    target: str = "clase_celda",
    hidden_layer_sizes: tuple = (8,),
    learning_rate_init: float = 0.01,
    alpha: float = 1e-3,
    activation: str = "relu",
    max_epocas: int = 300,
    features: Sequence[str] = FEATURES,
    seed: int = 42,
    registrar_curvas: bool = True,
) -> ResultadoEntrenamiento:
    """Entrena un MLP registrando la pérdida train/val por época.

    Usa ``partial_fit`` para poder medir log-loss (entropía cruzada) sobre los
    conjuntos de entrenamiento y validación en cada época, generando las curvas
    de aprendizaje pedidas por el taller.
    """
    Xtr, ytr = preparar_xy(train, target, features)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    clases = np.unique(ytr)

    modelo = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=1,
        warm_start=True,
        random_state=seed,
    )

    res = ResultadoEntrenamiento(
        modelo=modelo, scaler=scaler, clases=clases,
        config={
            "hidden_layer_sizes": hidden_layer_sizes,
            "learning_rate_init": learning_rate_init,
            "alpha": alpha,
            "activation": activation,
            "target": target,
        },
    )

    Xva = yva = None
    if val is not None and len(val) > 0:
        Xva, yva = preparar_xy(val, target, features)
        Xva = scaler.transform(Xva)

    import warnings
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(max_epocas):
            modelo.partial_fit(Xtr, ytr, classes=clases)
            if registrar_curvas:
                ptr = modelo.predict_proba(Xtr)
                res.loss_train.append(log_loss(ytr, ptr, labels=clases))
                res.acc_train.append(accuracy_score(ytr, modelo.predict(Xtr)))
                if Xva is not None:
                    pva = modelo.predict_proba(Xva)
                    res.loss_val.append(log_loss(yva, pva, labels=clases))
                    res.acc_val.append(accuracy_score(yva, modelo.predict(Xva)))
    return res


# ---------------------------------------------------------------------------
# Comparación de arquitecturas
# ---------------------------------------------------------------------------
def comparar_arquitecturas(
    particion: Particion,
    target: str = "clase_celda",
    configs: Optional[Sequence[dict]] = None,
    max_epocas: int = 300,
    features: Sequence[str] = FEATURES,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Entrena y compara varias arquitecturas de MLP.

    Returns
    -------
    (tabla, resultados):
        ``tabla`` es un DataFrame con métricas por configuración (loss/acc en
        train y val); ``resultados`` mapea la etiqueta de config al
        :class:`ResultadoEntrenamiento` (para graficar curvas).
    """
    if configs is None:
        configs = configuraciones_por_defecto()

    filas = []
    resultados = {}
    for cfg in configs:
        res = entrenar_mlp(
            particion.train,
            particion.val,
            target=target,
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            learning_rate_init=cfg["learning_rate_init"],
            alpha=cfg.get("alpha", 1e-3),
            activation=cfg.get("activation", "relu"),
            max_epocas=max_epocas,
            features=features,
            seed=seed,
        )
        etiqueta = _nombre_config(cfg)
        resultados[etiqueta] = res

        eval_df = particion.val if (particion.val is not None and len(particion.val)) else particion.test
        yv = eval_df[target].to_numpy()
        acc_eval = accuracy_score(yv, res.predecir(eval_df, features))
        filas.append({
            "config": etiqueta,
            "capas": len(cfg["hidden_layer_sizes"]),
            "arquitectura": str(cfg["hidden_layer_sizes"]),
            "lr": cfg["learning_rate_init"],
            "loss_train_final": res.loss_train[-1] if res.loss_train else np.nan,
            "loss_val_final": res.loss_val[-1] if res.loss_val else np.nan,
            "acc_train_final": res.acc_train[-1] if res.acc_train else np.nan,
            "acc_val_final": acc_eval,
        })
    tabla = pd.DataFrame(filas).sort_values("acc_val_final", ascending=False)
    return tabla.reset_index(drop=True), resultados


def configuraciones_por_defecto() -> list[dict]:
    """Rejilla mínima que cumple el taller: capas {1,2,3}, neuronas, lr {0.001,0.01,0.1}."""
    lrs = [0.001, 0.01, 0.1]
    arqs = [
        (8,), (16,),            # 1 capa, 2 configuraciones
        (16, 8), (32, 16),      # 2 capas, 2 configuraciones
        (32, 16, 8), (24, 16, 8),  # 3 capas, 2 configuraciones
    ]
    configs = []
    for arq in arqs:
        for lr in lrs:
            configs.append({
                "hidden_layer_sizes": arq,
                "learning_rate_init": lr,
                "alpha": 1e-3,
                "activation": "relu",
            })
    return configs


def _nombre_config(cfg: dict) -> str:
    arq = "x".join(str(h) for h in cfg["hidden_layer_sizes"])
    return f"L{len(cfg['hidden_layer_sizes'])}[{arq}]_lr{cfg['learning_rate_init']}"


# ---------------------------------------------------------------------------
# Comparación de funciones de pérdida: cross-entropy vs MSE
# ---------------------------------------------------------------------------
def comparar_perdidas(
    particion: Particion,
    target: str = "clase_celda",
    hidden_layer_sizes: tuple = (16, 8),
    learning_rate_init: float = 0.01,
    max_epocas: int = 300,
    features: Sequence[str] = FEATURES,
    seed: int = 42,
) -> pd.DataFrame:
    """Compara entropía cruzada (MLPClassifier) vs MSE (MLPRegressor one-hot).

    Para el criterio MSE se entrena un regresor sobre la codificación one-hot
    de las clases y se predice por ``argmax``; se reporta el MSE de validación
    y la exactitud resultante, frente al modelo de entropía cruzada.
    """
    # 1) Entropía cruzada.
    res_ce = entrenar_mlp(
        particion.train, particion.val, target=target,
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        max_epocas=max_epocas, features=features, seed=seed,
    )
    eval_df = particion.test
    yte = eval_df[target].to_numpy()
    acc_ce = accuracy_score(yte, res_ce.predecir(eval_df, features))

    # 2) MSE con MLPRegressor sobre one-hot.
    Xtr, ytr = preparar_xy(particion.train, target, features)
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    clases = np.unique(ytr)
    idx = {c: i for i, c in enumerate(clases)}
    Ytr = np.eye(len(clases))[[idx[v] for v in ytr]]

    reg = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes, activation="relu", solver="adam",
        learning_rate_init=learning_rate_init, max_iter=max_epocas,
        random_state=seed,
    )
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        reg.fit(Xtr, Ytr)

    Xte = scaler.transform(eval_df[list(features)].to_numpy(dtype=float))
    pred_mse = clases[np.argmax(reg.predict(Xte), axis=1)]
    acc_mse = accuracy_score(yte, pred_mse)
    Yte = np.eye(len(clases))[[idx.get(v, 0) for v in yte]]
    mse_val = mean_squared_error(Yte, reg.predict(Xte))

    return pd.DataFrame([
        {"perdida": "entropia_cruzada", "accuracy_test": acc_ce,
         "loss_final": res_ce.loss_train[-1] if res_ce.loss_train else np.nan},
        {"perdida": "mse", "accuracy_test": acc_mse, "loss_final": mse_val},
    ])


# ---------------------------------------------------------------------------
# Validación cruzada k-fold agrupada por simulación
# ---------------------------------------------------------------------------
def validacion_cruzada(
    datos: pd.DataFrame,
    target: str = "clase_celda",
    hidden_layer_sizes: tuple = (16, 8),
    learning_rate_init: float = 0.01,
    k: int = 5,
    max_epocas: int = 200,
    features: Sequence[str] = FEATURES,
    seed: int = 42,
) -> pd.DataFrame:
    """Validación cruzada k-fold **agrupada por simulación**.

    Se usa :class:`GroupKFold` con ``sim_id`` como grupo para que ninguna
    simulación aparezca simultáneamente en train y test (evita fuga).
    """
    col = filtrar_coloreadas(datos)
    X = col[list(features)].to_numpy(dtype=float)
    y = col[target].to_numpy()
    grupos = col["sim_id"].to_numpy()

    gkf = GroupKFold(n_splits=k)
    filas = []
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    for i, (tr_idx, te_idx) in enumerate(gkf.split(X, y, grupos), start=1):
        scaler = StandardScaler().fit(X[tr_idx])
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, activation="relu",
            solver="adam", learning_rate_init=learning_rate_init,
            max_iter=max_epocas, random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(scaler.transform(X[tr_idx]), y[tr_idx])
        acc = accuracy_score(y[te_idx], clf.predict(scaler.transform(X[te_idx])))
        filas.append({"fold": i, "n_test": len(te_idx), "accuracy": acc})
    tabla = pd.DataFrame(filas)
    tabla.attrs["media"] = tabla["accuracy"].mean()
    tabla.attrs["std"] = tabla["accuracy"].std()
    return tabla

"""Regresión logística como línea base interpretable.

Se ajustan modelos binario (coocurrente vs no_coocurrente) y multinomial
(3 clases) con ``statsmodels`` para obtener coeficientes, errores estándar,
valores p e intervalos de confianza al 95%. Se incluye además selección de
variables por AIC (backward stepwise) y regularización Lasso/Ridge.

Sirve de comparación con el MLP en términos de desempeño e interpretabilidad,
tal como pide el taller de referencia.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from .dataset import FEATURES, filtrar_coloreadas
from . import metrics


def _preparar(
    df: pd.DataFrame, target: str, features: Sequence[str], estandarizar: bool
):
    X = df[list(features)].to_numpy(dtype=float)
    if estandarizar:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    else:
        scaler = None
    y = df[target].to_numpy()
    return X, y, scaler


def logistica_binaria(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str = "clase_binaria",
    positivo: str = "coocurrente",
    features: Sequence[str] = FEATURES,
    estandarizar: bool = True,
) -> dict:
    """Ajusta regresión logística binaria e informa inferencia y métricas.

    Returns
    -------
    dict con:
        ``coeficientes`` (DataFrame: coef, SE, z, p, IC95), ``metricas``
        (dict binario en test), ``auc`` y el objeto ``modelo`` de statsmodels.
    """
    Xtr, ytr, scaler = _preparar(train, target, features, estandarizar)
    y_bin = (ytr == positivo).astype(int)
    Xc = sm.add_constant(Xtr, has_constant="add")
    nombres = ["const"] + list(features)
    negativo = [c for c in ["coocurrente", "no_coocurrente"] if c != positivo][0]

    Xte = scaler.transform(test[list(features)].to_numpy(dtype=float)) if scaler else test[list(features)].to_numpy(dtype=float)
    Xte_c = sm.add_constant(Xte, has_constant="add")

    modelo = None
    separacion = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = sm.Logit(y_bin, Xc).fit(disp=False, maxiter=200)
            _ = modelo.bse  # fuerza evaluación de errores estándar
        if not np.all(np.isfinite(np.asarray(modelo.bse))):
            raise np.linalg.LinAlgError("errores estándar no finitos")
        coefs = _tabla_coeficientes(modelo, nombres)
        proba = np.asarray(modelo.predict(Xte_c))
    except Exception:
        # Separación perfecta/cuasi-perfecta: la inferencia clásica no es
        # identificable. Se reportan coeficientes de una logística regularizada
        # (L2) y se anulan SE/p-valores/IC, marcando el hallazgo.
        separacion = True
        skl = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, y_bin)
        coef = np.concatenate([skl.intercept_, skl.coef_.ravel()])
        coefs = pd.DataFrame({
            "variable": nombres, "coef": coef,
            "std_err": np.nan, "z": np.nan, "p_valor": np.nan,
            "ic_inf": np.nan, "ic_sup": np.nan,
            "odds_ratio": np.exp(coef), "significativo": False,
        })
        proba = skl.predict_proba(Xte)[:, list(skl.classes_).index(1)]

    yte_bin = (test[target].to_numpy() == positivo).astype(int)
    pred_lbl = np.where(proba >= 0.5, positivo, negativo)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(yte_bin, proba) if len(np.unique(yte_bin)) > 1 else float("nan")
    met = metrics.metricas_binarias_detalle(test[target].to_numpy(), pred_lbl, positivo)
    youden = metrics.umbral_youden(yte_bin, proba) if len(np.unique(yte_bin)) > 1 else None

    return {
        "coeficientes": coefs,
        "metricas": met,
        "auc": auc,
        "youden": youden,
        "modelo": modelo,
        "separacion": separacion,
        "features": list(features),
    }


def logistica_multinomial(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str = "clase_celda",
    features: Sequence[str] = FEATURES,
    estandarizar: bool = True,
) -> dict:
    """Ajusta regresión logística multinomial (3 clases) con inferencia."""
    Xtr, ytr, scaler = _preparar(train, target, features, estandarizar)
    clases, y_idx = np.unique(ytr, return_inverse=True)
    Xc = sm.add_constant(Xtr, has_constant="add")
    Xte = scaler.transform(test[list(features)].to_numpy(dtype=float)) if scaler else test[list(features)].to_numpy(dtype=float)
    Xte_c = sm.add_constant(Xte, has_constant="add")

    separacion = False
    resumen_txt = ""
    modelo = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = sm.MNLogit(y_idx, Xc).fit(disp=False, maxiter=200, method="bfgs")
        proba = np.asarray(modelo.predict(Xte_c))
        resumen_txt = modelo.summary().as_text()
    except Exception:
        separacion = True
        skl = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, ytr)
        # Alinear columnas de proba al orden de `clases`.
        proba_skl = skl.predict_proba(Xte)
        orden = [list(skl.classes_).index(c) for c in clases]
        proba = proba_skl[:, orden]
        resumen_txt = "Ajuste multinomial por separación: se usó LogisticRegression (L2). Inferencia clásica no disponible."

    pred_idx = np.argmax(proba, axis=1)
    pred_lbl = clases[pred_idx]
    rep = metrics.evaluar(
        test[target].to_numpy(), pred_lbl, y_proba=proba, clases=list(clases)
    )
    return {
        "coeficientes_resumen": resumen_txt,
        "reporte": rep,
        "clases": list(clases),
        "modelo": modelo,
        "separacion": separacion,
    }


def _tabla_coeficientes(modelo, nombres: Sequence[str]) -> pd.DataFrame:
    """Extrae coef, SE, z, p e IC95 de un modelo statsmodels ajustado."""
    ci = modelo.conf_int()
    ci = np.asarray(ci)
    tabla = pd.DataFrame({
        "variable": nombres,
        "coef": np.asarray(modelo.params).ravel(),
        "std_err": np.asarray(modelo.bse).ravel(),
        "z": np.asarray(modelo.tvalues).ravel(),
        "p_valor": np.asarray(modelo.pvalues).ravel(),
        "ic_inf": ci[:, 0],
        "ic_sup": ci[:, 1],
    })
    tabla["odds_ratio"] = np.exp(tabla["coef"])
    tabla["significativo"] = tabla["p_valor"] < 0.05
    return tabla


def seleccion_backward_aic(
    train: pd.DataFrame,
    target: str = "clase_binaria",
    positivo: str = "coocurrente",
    features: Sequence[str] = FEATURES,
) -> dict:
    """Selección de variables backward por AIC sobre la logística binaria.

    Elimina iterativamente la variable cuya remoción más reduce el AIC hasta
    que ninguna remoción lo mejore.
    """
    Xtr, ytr, scaler = _preparar(train, target, list(features), True)
    y_bin = (ytr == positivo).astype(int)
    df_X = pd.DataFrame(Xtr, columns=list(features))

    seleccion = list(features)

    def aic_de(cols):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Xc = sm.add_constant(df_X[cols].to_numpy(), has_constant="add")
                m = sm.Logit(y_bin, Xc).fit(disp=False, maxiter=200)
            return float(m.aic) if np.isfinite(m.aic) else float("inf")
        except Exception:
            return float("inf")

    aic_actual = aic_de(seleccion)
    historia = [{"paso": 0, "variables": list(seleccion), "aic": aic_actual}]
    paso = 0
    mejorando = True
    while mejorando and len(seleccion) > 1:
        mejorando = False
        mejor_aic = aic_actual
        peor = None
        for c in seleccion:
            candidato = [x for x in seleccion if x != c]
            aic_c = aic_de(candidato)
            if aic_c < mejor_aic:
                mejor_aic = aic_c
                peor = c
        if peor is not None:
            seleccion.remove(peor)
            aic_actual = mejor_aic
            paso += 1
            historia.append({"paso": paso, "eliminada": peor,
                             "variables": list(seleccion), "aic": aic_actual})
            mejorando = True

    return {"seleccion_final": seleccion, "historia": pd.DataFrame(historia),
            "aic_final": aic_actual}


def seleccion_lasso(
    train: pd.DataFrame,
    target: str = "clase_binaria",
    positivo: str = "coocurrente",
    features: Sequence[str] = FEATURES,
    penalty: str = "l1",
    seed: int = 42,
) -> pd.DataFrame:
    """Regularización Lasso (L1) o Ridge (L2) con validación cruzada.

    Devuelve los coeficientes por variable (los cercanos a 0 en Lasso indican
    variables descartadas).
    """
    Xtr, ytr, _ = _preparar(train, target, list(features), True)
    y_bin = (ytr == positivo).astype(int)
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    modelo = LogisticRegressionCV(
        Cs=10, cv=5, penalty=penalty, solver=solver, max_iter=500,
        random_state=seed,
    ).fit(Xtr, y_bin)
    coefs = modelo.coef_.ravel()
    tabla = pd.DataFrame({
        "variable": list(features),
        "coef": coefs,
        "abs_coef": np.abs(coefs),
        "seleccionada": np.abs(coefs) > 1e-6,
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    tabla.attrs["C_optimo"] = float(modelo.C_[0])
    tabla.attrs["penalty"] = penalty
    return tabla

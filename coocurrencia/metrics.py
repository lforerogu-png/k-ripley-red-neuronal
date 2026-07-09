"""Métricas de evaluación **condicionadas** a celdas coloreadas.

Todas las funciones asumen que ``y_true`` / ``y_pred`` provienen del conjunto
ya filtrado (sin celdas blancas). Se ofrecen:

* Matriz de confusión 3x3 (atraccion / repulsion / sin_coocurrencia).
* Matriz de confusión 2x2 (coocurrente / no_coocurrente).
* Reporte completo: accuracy, precisión/recall/F1 por clase, macro y weighted.
* ROC-AUC uno-contra-todos (OvR) multiclase y binario.
* Conteo informativo de celdas blancas (no entra en las métricas del modelo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from . import CLASES_2, CLASES_3


def matriz_confusion(
    y_true: Sequence, y_pred: Sequence, etiquetas: Sequence[str]
) -> pd.DataFrame:
    """Matriz de confusión como DataFrame (filas=real, columnas=predicho)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(etiquetas))
    return pd.DataFrame(
        cm,
        index=[f"real_{e}" for e in etiquetas],
        columns=[f"pred_{e}" for e in etiquetas],
    )


def colapsar_binaria(
    etiquetas_3: Sequence[str], criterio: str = "colapso"
) -> np.ndarray:
    """Colapsa etiquetas de 3 clases a binario.

    ``"colapso"``: {atraccion, repulsion} -> coocurrente; sin_coocurrencia ->
    no_coocurrente (equivalente al R original).
    """
    out = []
    for e in etiquetas_3:
        if criterio == "colapso":
            out.append(
                "coocurrente" if e in ("atraccion", "repulsion") else "no_coocurrente"
            )
        else:
            out.append(e)
    return np.array(out)


@dataclass
class ReporteClasificacion:
    """Contenedor de métricas de un clasificador sobre el espacio condicional."""

    accuracy: float
    por_clase: pd.DataFrame          # precision/recall/f1/support por clase
    macro: dict
    weighted: dict
    matriz_3x3: Optional[pd.DataFrame]
    matriz_2x2: Optional[pd.DataFrame]
    auc_ovr: Optional[dict]
    n_evaluadas: int
    n_blancas: Optional[int] = None  # informativo, NO entra en métricas

    def resumen(self) -> str:
        lineas = [
            f"Accuracy (condicional): {self.accuracy:.4f}",
            f"Celdas evaluadas (coloreadas): {self.n_evaluadas}",
        ]
        if self.n_blancas is not None:
            lineas.append(
                f"Celdas blancas (informativo, excluidas): {self.n_blancas}"
            )
        lineas.append(
            "Macro F1: {:.4f} | Weighted F1: {:.4f}".format(
                self.macro["f1"], self.weighted["f1"]
            )
        )
        return "\n".join(lineas)


def evaluar(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_proba: Optional[np.ndarray] = None,
    clases: Sequence[str] = CLASES_3,
    n_blancas: Optional[int] = None,
    criterio_binario: str = "colapso",
) -> ReporteClasificacion:
    """Calcula el reporte completo sobre el espacio condicional.

    Parameters
    ----------
    y_true, y_pred:
        Etiquetas verdaderas y predichas (solo celdas coloreadas).
    y_proba:
        Matriz de probabilidades ``(n, n_clases)`` alineada con ``clases``
        para el AUC-ROC (opcional).
    clases:
        Orden de clases (3 o 2).
    n_blancas:
        Número de celdas blancas del conjunto (informativo).
    criterio_binario:
        Criterio de colapso a 2x2 cuando ``clases`` tiene 3 elementos.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(clases), zero_division=0
    )
    por_clase = pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": s}, index=list(clases)
    )
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(clases), average="macro", zero_division=0
    )
    wp, wr, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(clases), average="weighted", zero_division=0
    )
    macro = {"precision": mp, "recall": mr, "f1": mf}
    weighted = {"precision": wp, "recall": wr, "f1": wf}

    es_tres = len(clases) == 3
    m3 = matriz_confusion(y_true, y_pred, clases) if es_tres else None
    if es_tres:
        yt2 = colapsar_binaria(y_true, criterio="colapso")
        yp2 = colapsar_binaria(y_pred, criterio="colapso")
        m2 = matriz_confusion(yt2, yp2, CLASES_2)
    else:
        m2 = matriz_confusion(y_true, y_pred, clases)

    auc = _auc_ovr(y_true, y_proba, clases) if y_proba is not None else None

    return ReporteClasificacion(
        accuracy=acc,
        por_clase=por_clase,
        macro=macro,
        weighted=weighted,
        matriz_3x3=m3,
        matriz_2x2=m2,
        auc_ovr=auc,
        n_evaluadas=len(y_true),
        n_blancas=n_blancas,
    )


def _auc_ovr(
    y_true: np.ndarray, y_proba: np.ndarray, clases: Sequence[str]
) -> Optional[dict]:
    """AUC-ROC uno-contra-todos por clase y promedio macro."""
    try:
        clases = list(clases)
        presentes = set(np.unique(y_true))
        aucs = {}
        for i, c in enumerate(clases):
            if c not in presentes:
                aucs[c] = float("nan")
                continue
            y_bin = (y_true == c).astype(int)
            if len(np.unique(y_bin)) < 2:
                aucs[c] = float("nan")
                continue
            aucs[c] = roc_auc_score(y_bin, y_proba[:, i])
        vals = [v for v in aucs.values() if np.isfinite(v)]
        aucs["macro"] = float(np.mean(vals)) if vals else float("nan")
        return aucs
    except Exception:
        return None


def metricas_binarias_detalle(
    y_true: Sequence[str], y_pred: Sequence[str], positivo: str = "coocurrente"
) -> dict:
    """Sensibilidad, especificidad, precisión y F1 para el problema binario."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    negativo = [c for c in CLASES_2 if c != positivo][0]
    cm = confusion_matrix(y_true, y_pred, labels=[positivo, negativo])
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]
    n = cm.sum()
    return {
        "accuracy": (tp + tn) / n if n else float("nan"),
        "sensibilidad": tp / (tp + fn) if (tp + fn) else float("nan"),
        "especificidad": tn / (tn + fp) if (tn + fp) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"),
    }


def accuracy_condicional(
    df_test: pd.DataFrame, resultado_entrenamiento, target: str = "clase_celda"
) -> float:
    """Exactitud del modelo sobre las celdas coloreadas de ``df_test``."""
    pred = resultado_entrenamiento.predecir(df_test)
    return accuracy_score(df_test[target].to_numpy(), pred)


def umbral_youden(y_true_bin: np.ndarray, y_score: np.ndarray) -> dict:
    """Umbral óptimo por índice de Youden (J = sensibilidad + especificidad - 1)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_true_bin, y_score)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return {
        "umbral": float(thr[idx]),
        "sensibilidad": float(tpr[idx]),
        "especificidad": float(1 - fpr[idx]),
        "j": float(j[idx]),
    }

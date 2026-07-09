"""Construcción del conjunto de datos a nivel de celda (espacio condicional).

Este módulo materializa la corrección metodológica central del proyecto.

Para cada celda de la grilla se generan las variables:

* ``ocupa_p1``, ``ocupa_p2`` (binarias)
* ``k1_val``, ``k2_val``, ``k12_val`` y sus versiones normalizadas ``*_n``
* ``cls1_f``, ``cls2_f`` (clases univariadas codificadas)
* ``clase_celda`` (etiqueta de 3 clases) y ``clase_binaria`` (2 clases)

Reglas de etiquetado (siguiendo al profesor en la reunión, min. 30-37):

* ``ocupa_p1 == 1`` y ``ocupa_p2 == 1`` (celda **verde**)  -> clase bivariada
  real de K12: ``atraccion`` / ``repulsion`` / ``sin_coocurrencia``.
* solo uno ocupa (celda **azul** o **amarilla**) -> ``repulsion``
  (evento único: "cuando aparece solo un color... esa sería la categoría
  repulsiva", min. 35:35).
* ninguno ocupa (celda **blanca**) -> se marca ``celda_blanca`` y se
  **excluye** de todo cómputo de métricas del modelo.

Etiqueta binaria (``clase_binaria``): por defecto se usa el criterio de
**ocupación** del profesor (min. 37): ``coocurrente`` = celda verde (ambos
eventos presentes); ``no_coocurrente`` = todo lo demás coloreado. Se ofrece
también el criterio de **colapso** de la etiqueta de 3 clases
(atraccion+repulsion -> coocurrente) equivalente al script R original.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import ripley
from .patterns import generar_patron, generar_patron_condicionado

_MAP_UNIV = {"agrupado": 1, "aleatorio": 2, "disperso": 3}


@dataclass
class ResumenPar:
    """Estadísticos de K de un par de patrones (A, B)."""

    k1_val: float
    k2_val: float
    k12_val: float
    cls1: str
    cls2: str
    clase_biv: str
    k1_curva: ripley.ResultadoK = field(repr=False, default=None)
    k2_curva: ripley.ResultadoK = field(repr=False, default=None)
    k12_curva: ripley.ResultadoK = field(repr=False, default=None)


def analizar_par(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
    r_ref: float = 0.2,
    metodo: str = "umbral",
    *,
    tol: float = 0.15,
    n_sim_env: int = 39,
    seed_env: Optional[int] = 12345,
) -> ResumenPar:
    """Calcula K1, K2, K12 y las clasifica (por umbral o envolvente).

    Parameters
    ----------
    p1, p2:
        Patrones A y B (con columnas ``x``, ``y``).
    r_ref:
        Radio de referencia para extraer el valor de K y clasificar.
    metodo:
        ``"umbral"`` (comparación con ``pi*r^2``) o ``"envolvente"``
        (Monte Carlo bajo CSR / independencia).
    tol:
        Tolerancia relativa del método de umbral.
    n_sim_env:
        Número de simulaciones para las envolventes.
    seed_env:
        Semilla base de las envolventes.
    """
    a = p1[["x", "y"]].to_numpy() if len(p1) else np.empty((0, 2))
    b = p2[["x", "y"]].to_numpy() if len(p2) else np.empty((0, 2))

    k1 = ripley.k_univariada(a)
    k2 = ripley.k_univariada(b)
    k12 = ripley.k_cruzada(a, b)

    k1_val = k1.valor_en(r_ref)
    k2_val = k2.valor_en(r_ref)
    k12_val = k12.valor_en(r_ref)

    if metodo == "envolvente":
        env1 = ripley.envolvente_csr(len(a), n_sim=n_sim_env, seed=seed_env)
        env2 = ripley.envolvente_csr(len(b), n_sim=n_sim_env, seed=seed_env)
        cls1 = ripley.clasificar_por_envolvente(k1, env1, r_ref, "univariada")
        cls2 = ripley.clasificar_por_envolvente(k2, env2, r_ref, "univariada")
        if len(a) and len(b):
            env12 = ripley.envolvente_independencia(
                a, b, n_sim=n_sim_env, seed=seed_env
            )
            clase_biv = ripley.clasificar_por_envolvente(
                k12, env12, r_ref, "bivariada"
            )
        else:
            clase_biv = "sin_coocurrencia"
    else:
        cls1 = ripley.clasificar_univariada_umbral(k1, r_ref, tol)
        cls2 = ripley.clasificar_univariada_umbral(k2, r_ref, tol)
        clase_biv = ripley.clasificar_bivariada_umbral(k12, r_ref, tol)

    return ResumenPar(
        k1_val=k1_val, k2_val=k2_val, k12_val=k12_val,
        cls1=cls1, cls2=cls2, clase_biv=clase_biv,
        k1_curva=k1, k2_curva=k2, k12_curva=k12,
    )


def etiquetar_celda(ocupa_p1: bool, ocupa_p2: bool, clase_biv: str) -> str:
    """Etiqueta de 3 clases de una celda coloreada.

    * verde (ambos)        -> ``clase_biv`` (atraccion/repulsion/sin_coocurrencia)
    * azul o amarillo (uno) -> ``repulsion`` (evento único)
    """
    if ocupa_p1 and ocupa_p2:
        return clase_biv
    return "repulsion"


def etiqueta_binaria(
    ocupa_p1: bool, ocupa_p2: bool, clase_3: str, criterio: str = "ocupacion"
) -> str:
    """Etiqueta binaria de una celda coloreada.

    Parameters
    ----------
    criterio:
        ``"ocupacion"`` (profesor min. 37): coocurrente = ambos presentes;
        ``"colapso"`` (R original): coocurrente = clase_3 en {atraccion, repulsion}.
    """
    if criterio == "colapso":
        return "coocurrente" if clase_3 in ("atraccion", "repulsion") else "no_coocurrente"
    return "coocurrente" if (ocupa_p1 and ocupa_p2) else "no_coocurrente"


def construir_datos_celda(
    p1: pd.DataFrame,
    p2: pd.DataFrame,
    resumen: ResumenPar,
    n_grid: int = 30,
    k_max: Optional[float] = None,
    criterio_binario: str = "ocupacion",
) -> pd.DataFrame:
    """Construye el DataFrame a nivel de celda para un par de patrones.

    Returns
    -------
    pandas.DataFrame
        Una fila por celda de la grilla (``n_grid**2`` filas) con features,
        indicador ``celda_blanca`` y etiquetas ``clase_celda`` y
        ``clase_binaria``.
    """
    cols, rows = np.meshgrid(np.arange(1, n_grid + 1), np.arange(1, n_grid + 1))
    grilla = pd.DataFrame({"col": cols.ravel(), "row": rows.ravel()})

    s1 = set(zip(p1["col"], p1["row"])) if len(p1) else set()
    s2 = set(zip(p2["col"], p2["row"])) if len(p2) else set()
    ocupa1 = np.array([(c, r) in s1 for c, r in zip(grilla["col"], grilla["row"])])
    ocupa2 = np.array([(c, r) in s2 for c, r in zip(grilla["col"], grilla["row"])])

    grilla["ocupa_p1"] = ocupa1.astype(int)
    grilla["ocupa_p2"] = ocupa2.astype(int)
    grilla["celda_blanca"] = (~ocupa1) & (~ocupa2)

    grilla["k1_val"] = resumen.k1_val
    grilla["k2_val"] = resumen.k2_val
    grilla["k12_val"] = resumen.k12_val
    grilla["cls1_f"] = _MAP_UNIV.get(resumen.cls1, 2)
    grilla["cls2_f"] = _MAP_UNIV.get(resumen.cls2, 2)
    grilla["cls1"] = resumen.cls1
    grilla["cls2"] = resumen.cls2

    if k_max is not None and np.isfinite(k_max) and k_max > 0:
        grilla["k1_n"] = resumen.k1_val / k_max
        grilla["k2_n"] = resumen.k2_val / k_max
        grilla["k12_n"] = resumen.k12_val / k_max

    clase3 = [
        etiquetar_celda(bool(o1), bool(o2), resumen.clase_biv)
        for o1, o2 in zip(ocupa1, ocupa2)
    ]
    grilla["clase_celda"] = clase3
    grilla["clase_binaria"] = [
        etiqueta_binaria(bool(o1), bool(o2), c3, criterio_binario)
        for o1, o2, c3 in zip(ocupa1, ocupa2, clase3)
    ]
    # Las celdas blancas no tienen etiqueta válida para el modelo.
    grilla.loc[grilla["celda_blanca"], ["clase_celda", "clase_binaria"]] = np.nan
    return grilla


def simular_dataset(
    n_sim: int = 200,
    n_grid: int = 30,
    n_pts_min: int = 60,
    n_pts_max: int = 200,
    r_ref: float = 0.2,
    metodo: str = "umbral",
    criterio_binario: str = "ocupacion",
    n_coincidentes_rango: Optional[tuple[int, int]] = None,
    seed: int = 42,
    n_sim_env: int = 39,
    verbose: bool = False,
    p1_fijo: Optional[pd.DataFrame] = None,
    p2_fijo: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Simula ``n_sim`` pares de patrones y arma el dataset completo.

    Cada simulación elige tipos de A y B al azar, un número de puntos y un
    número de celdas coincidentes (para forzar coocurrencia controlada), y
    calcula sus K. El resultado es un DataFrame concatenado con ``sim_id``.

    Se añade normalización global ``k*_n`` y se expone ``k_max`` como atributo
    ``df.attrs['k_max']``.

    Parameters
    ----------
    metodo:
        Método de etiquetado de K (``"umbral"`` o ``"envolvente"``).
    criterio_binario:
        Criterio para ``clase_binaria`` (``"ocupacion"`` o ``"colapso"``).
    n_coincidentes_rango:
        Rango (min, max) de celdas coincidentes forzadas; por defecto se elige
        entre 0 y min(n_pts, n_A).
    p1_fijo:
        Si se provee (por ejemplo, puntos reales cargados desde un CSV), se
        usa este patrón A **fijo** en todas las simulaciones en lugar de
        generarlo al azar; solo el patrón B varía entre simulaciones.
    p2_fijo:
        Igual que ``p1_fijo`` pero para el patrón B: si se provee, se usa
        fijo en todas las simulaciones en lugar de generarlo condicionado a
        A. Si ambos (``p1_fijo`` y ``p2_fijo``) se proveen, el par queda
        completamente fijo y todas las simulaciones producen el mismo
        resultado (no hay variabilidad para entrenar, pero permite evaluar
        el par real bajo el mismo pipeline).
    """
    tipos = ("agrupado", "disperso", "aleatorio")
    rng = np.random.default_rng(seed)
    bloques: list[pd.DataFrame] = []
    resumenes: list[ResumenPar] = []

    intentos = 0
    ok = 0
    while ok < n_sim and intentos < n_sim * 5:
        intentos += 1
        s = int(rng.integers(1, 10_000_000))
        t2 = tipos[rng.integers(3)]
        n_pts = int(rng.integers(n_pts_min, n_pts_max + 1))

        if p1_fijo is not None:
            p1 = p1_fijo
        else:
            t1 = tipos[rng.integers(3)]
            p1 = generar_patron(t1, n_pts, s, n_grid=n_grid)
        if len(p1) == 0:
            continue

        if p2_fijo is not None:
            p2 = p2_fijo
        else:
            max_coinc = min(len(p1), n_pts)
            if n_coincidentes_rango is None:
                n_coinc = int(rng.integers(0, max_coinc + 1))
            else:
                lo, hi = n_coincidentes_rango
                n_coinc = int(rng.integers(lo, min(hi, max_coinc) + 1))
            p2 = generar_patron_condicionado(
                p1, t2, n_pts, n_coinc, s + 500, n_grid=n_grid
            )
        if len(p2) == 0:
            continue

        resumen = analizar_par(
            p1, p2, r_ref, metodo, n_sim_env=n_sim_env, seed_env=s + 900
        )
        bloque = construir_datos_celda(
            p1, p2, resumen, n_grid=n_grid, criterio_binario=criterio_binario
        )
        ok += 1
        bloque["sim_id"] = ok
        bloques.append(bloque)
        resumenes.append(resumen)
        if verbose and ok % 25 == 0:
            print(f"  simulaciones completadas: {ok}/{n_sim}")

    if not bloques:
        return pd.DataFrame()

    datos = pd.concat(bloques, ignore_index=True)
    k_max = float(
        np.nanmax(
            np.concatenate(
                [datos["k1_val"], datos["k2_val"], datos["k12_val"]]
            )
        )
        + 1e-9
    )
    datos["k1_n"] = datos["k1_val"] / k_max
    datos["k2_n"] = datos["k2_val"] / k_max
    datos["k12_n"] = datos["k12_val"] / k_max
    datos.attrs["k_max"] = k_max
    datos.attrs["metodo"] = metodo
    datos.attrs["criterio_binario"] = criterio_binario
    return datos


def filtrar_coloreadas(datos: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo las celdas coloreadas (excluye blancas).

    Esta es la operación que implementa la corrección del profesor: el
    espacio muestral de evaluación queda condicionado a celdas con evento.
    """
    return datos.loc[~datos["celda_blanca"]].reset_index(drop=True)


# Variables de entrada estándar del modelo (a nivel de celda).
FEATURES = ["k1_n", "k2_n", "k12_n", "cls1_f", "cls2_f", "ocupa_p1", "ocupa_p2"]

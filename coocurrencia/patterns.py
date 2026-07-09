"""Generación de patrones de puntos espaciales sobre una grilla regular NxN.

Porta la lógica del script R original (``generar_patron`` y
``generar_patron2_con_coincidencia``) a Python, con dos mejoras pedidas:

* El número de puntos ya no está limitado a 150 (se admite 1..500).
* El tamaño de la grilla es parametrizable (default 30x30 = 900 celdas).
* Reproducibilidad estricta con ``numpy.random.default_rng(seed)``.

Cada patrón se representa como un :class:`pandas.DataFrame` con columnas
``col``, ``row`` (índices de celda, base 1) y ``x``, ``y`` (coordenadas
normalizadas al cuadrado unitario, como en spatstat).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TiposPatron = ("agrupado", "disperso", "aleatorio")


@dataclass(frozen=True)
class ParametrosPatron:
    """Parámetros de un patrón de puntos.

    Parameters
    ----------
    tipo:
        Uno de ``"agrupado"``, ``"disperso"`` o ``"aleatorio"``.
    n_puntos:
        Número de puntos objetivo (1..500).
    seed:
        Semilla para reproducibilidad.
    """

    tipo: str
    n_puntos: int
    seed: int


def _celdas_grilla(n_grid: int) -> pd.DataFrame:
    """Devuelve todas las celdas ``(col, row)`` de una grilla ``n_grid`` x ``n_grid``."""
    cols, rows = np.meshgrid(np.arange(1, n_grid + 1), np.arange(1, n_grid + 1))
    return pd.DataFrame({"col": cols.ravel(), "row": rows.ravel()})


def _añadir_coords(puntos: pd.DataFrame, n_grid: int) -> pd.DataFrame:
    """Añade coordenadas normalizadas ``x, y`` en (0, 1) al centro de cada celda."""
    puntos = puntos.copy()
    if len(puntos) == 0:
        puntos["x"] = pd.Series(dtype=float)
        puntos["y"] = pd.Series(dtype=float)
        return puntos
    puntos["x"] = (puntos["col"] - 0.5) / n_grid
    puntos["y"] = (puntos["row"] - 0.5) / n_grid
    return puntos


def generar_patron(
    tipo: str,
    n_puntos: int = 60,
    seed: Optional[int] = None,
    *,
    n_grid: int = 30,
    celdas_ocupadas: Optional[pd.DataFrame] = None,
    radio_cluster: int = 6,
    n_clusters: int = 4,
    dist_min_disperso: float = 3.0,
) -> pd.DataFrame:
    """Genera un patrón de puntos sobre la grilla.

    Parameters
    ----------
    tipo:
        ``"agrupado"`` (varios clústeres), ``"disperso"`` (inhibición por
        distancia mínima) o ``"aleatorio"`` (CSR: muestreo uniforme de celdas).
    n_puntos:
        Número de puntos deseado (1..500). No se limita a 150.
    seed:
        Semilla del generador ``numpy.random.default_rng``.
    n_grid:
        Tamaño de la grilla (n_grid x n_grid).
    celdas_ocupadas:
        Celdas ya ocupadas (por ejemplo por el patrón A) que deben excluirse
        al generar los puntos "libres" del patrón B.
    radio_cluster:
        Radio (en celdas) de dispersión alrededor de cada centro de clúster.
    n_clusters:
        Número de clústeres para el tipo ``"agrupado"``.
    dist_min_disperso:
        Distancia mínima (en celdas) entre puntos para el tipo ``"disperso"``.

    Returns
    -------
    pandas.DataFrame
        Columnas ``col``, ``row``, ``x``, ``y``. Puede tener menos de
        ``n_puntos`` filas si el espacio disponible no lo permite.
    """
    if tipo not in TiposPatron:
        raise ValueError(f"tipo debe ser uno de {TiposPatron}, no {tipo!r}")
    n_puntos = int(max(0, n_puntos))
    rng = np.random.default_rng(seed)

    celdas = _celdas_grilla(n_grid)
    if celdas_ocupadas is not None and len(celdas_ocupadas) > 0:
        ocupadas = set(zip(celdas_ocupadas["col"], celdas_ocupadas["row"]))
        mask = [
            (c, r) not in ocupadas
            for c, r in zip(celdas["col"], celdas["row"])
        ]
        celdas = celdas.loc[mask].reset_index(drop=True)

    if len(celdas) == 0 or n_puntos == 0:
        return _añadir_coords(celdas.iloc[0:0], n_grid)

    if tipo == "agrupado":
        puntos = _generar_agrupado(
            celdas, n_puntos, rng, n_clusters, radio_cluster
        )
    elif tipo == "disperso":
        puntos = _generar_disperso(celdas, n_puntos, rng, dist_min_disperso)
    else:  # aleatorio (CSR)
        n_sel = min(n_puntos, len(celdas))
        idx = rng.choice(len(celdas), size=n_sel, replace=False)
        puntos = celdas.iloc[idx].reset_index(drop=True)

    return _añadir_coords(puntos, n_grid)


def _generar_agrupado(
    celdas: pd.DataFrame,
    n_puntos: int,
    rng: np.random.Generator,
    n_clusters: int,
    radio_cluster: int,
) -> pd.DataFrame:
    """Patrón agrupado: puntos alrededor de ``n_clusters`` centros aleatorios."""
    n_clusters = int(min(n_clusters, len(celdas)))
    centros_idx = rng.choice(len(celdas), size=n_clusters, replace=False)
    centros = celdas.iloc[centros_idx]

    partes = []
    n_por_cluster = max(1, round(n_puntos / n_clusters))
    for _, centro in centros.iterrows():
        cx, cy = centro["col"], centro["row"]
        cand = celdas[
            (np.abs(celdas["col"] - cx) <= radio_cluster)
            & (np.abs(celdas["row"] - cy) <= radio_cluster)
        ]
        if len(cand) < n_por_cluster:
            cand = celdas
        n_take = min(n_por_cluster, len(cand))
        sel = cand.iloc[rng.choice(len(cand), size=n_take, replace=False)]
        partes.append(sel)

    puntos = pd.concat(partes, ignore_index=True)
    puntos = puntos.drop_duplicates(subset=["col", "row"]).reset_index(drop=True)
    if len(puntos) > n_puntos:
        puntos = puntos.iloc[:n_puntos].reset_index(drop=True)
    return puntos


def _generar_disperso(
    celdas: pd.DataFrame,
    n_puntos: int,
    rng: np.random.Generator,
    dist_min: float,
    max_intentos: int = 20000,
) -> pd.DataFrame:
    """Patrón disperso: proceso de inhibición simple (distancia mínima)."""
    coords = celdas[["col", "row"]].to_numpy()
    primero = rng.integers(len(coords))
    seleccion = [coords[primero]]
    intentos = 0
    while len(seleccion) < n_puntos and intentos < max_intentos:
        cand = coords[rng.integers(len(coords))]
        sel_arr = np.asarray(seleccion)
        ya = np.any((sel_arr[:, 0] == cand[0]) & (sel_arr[:, 1] == cand[1]))
        dists = np.sqrt(np.sum((sel_arr - cand) ** 2, axis=1))
        if not ya and dists.min() >= dist_min:
            seleccion.append(cand)
        intentos += 1
    arr = np.asarray(seleccion)
    return pd.DataFrame({"col": arr[:, 0], "row": arr[:, 1]})


def generar_patron_condicionado(
    p1: pd.DataFrame,
    tipo2: str,
    n_puntos: int,
    n_coincidentes: int,
    seed: Optional[int] = None,
    *,
    n_grid: int = 30,
    **kwargs,
) -> pd.DataFrame:
    """Genera el patrón B **condicionado** al patrón A.

    En vez de generar A y B de forma independiente (lo que producía muy pocas
    coocurrencias, como señala el profesor en la reunión), se fuerza que
    exactamente ``n_coincidentes`` celdas de B coincidan con celdas de A. El
    resto de puntos de B se generan según ``tipo2`` sobre celdas libres.

    Parameters
    ----------
    p1:
        Patrón A (DataFrame con ``col``, ``row``).
    tipo2:
        Tipo del patrón B (``agrupado``/``disperso``/``aleatorio``).
    n_puntos:
        Número total de puntos de B.
    n_coincidentes:
        Número de celdas de B que deben coincidir con A
        (0 .. min(len(p1), n_puntos)).
    seed:
        Semilla del generador.
    n_grid:
        Tamaño de la grilla.

    Returns
    -------
    pandas.DataFrame
        Patrón B con columnas ``col``, ``row``, ``x``, ``y``.
    """
    rng = np.random.default_rng(seed)
    n_puntos = int(max(0, n_puntos))
    n_coincidentes = int(
        max(0, min(n_coincidentes, len(p1), n_puntos))
    )

    if n_coincidentes > 0 and len(p1) > 0:
        idx = rng.choice(len(p1), size=n_coincidentes, replace=False)
        coinc = p1.iloc[idx][["col", "row"]].reset_index(drop=True)
    else:
        coinc = pd.DataFrame({"col": [], "row": []}).astype(
            {"col": int, "row": int}
        )

    n_restantes = n_puntos - n_coincidentes
    if n_restantes > 0:
        # Seed derivada para que los puntos libres no repliquen la selección de coincidentes.
        seed_libres = None if seed is None else int(seed) + 7919
        libres = generar_patron(
            tipo2,
            n_restantes,
            seed_libres,
            n_grid=n_grid,
            celdas_ocupadas=coinc,
            **kwargs,
        )
        p2 = pd.concat(
            [coinc, libres[["col", "row"]]], ignore_index=True
        ).drop_duplicates(subset=["col", "row"]).reset_index(drop=True)
    else:
        p2 = coinc

    return _añadir_coords(p2, n_grid)


def contar_coincidencias(p1: pd.DataFrame, p2: pd.DataFrame) -> int:
    """Cuenta celdas ocupadas simultáneamente por A y B (coocurrencia real)."""
    if len(p1) == 0 or len(p2) == 0:
        return 0
    s1 = set(zip(p1["col"], p1["row"]))
    s2 = set(zip(p2["col"], p2["row"]))
    return len(s1 & s2)

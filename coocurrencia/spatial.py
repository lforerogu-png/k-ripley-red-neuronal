"""Análisis de dependencia espacial: Índice de Moran global y local (LISA).

Se implementa desde cero (sin ``libpysal``/``esda``) para no añadir
dependencias frágiles y controlar el cálculo. Conecta con la preocupación del
profesor por la "coherencia" espacial de los eventos (epidemiología de
patrones de puntos, cólera, enfermedades en cultivos).

El Índice de Moran mide autocorrelación espacial:

.. math::

    I = \\frac{n}{\\sum_{i}\\sum_{j} w_{ij}}
        \\frac{\\sum_i \\sum_j w_{ij} (x_i-\\bar x)(x_j-\\bar x)}
             {\\sum_i (x_i-\\bar x)^2}

con significancia evaluada por permutaciones aleatorias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def matriz_pesos_grilla(
    n_grid: int, criterio: str = "queen", estandarizar: bool = True
) -> np.ndarray:
    """Matriz de pesos espaciales de contigüidad para una grilla ``n_grid``.

    Parameters
    ----------
    criterio:
        ``"queen"`` (8 vecinos) o ``"rook"`` (4 vecinos).
    estandarizar:
        Si ``True``, estandariza por filas (cada fila suma 1).

    Returns
    -------
    numpy.ndarray
        Matriz ``(n_grid**2, n_grid**2)`` en orden fila-mayor de la grilla
        ``(row, col)``.
    """
    n = n_grid * n_grid
    W = np.zeros((n, n))

    def idx(r, c):
        return r * n_grid + c

    for r in range(n_grid):
        for c in range(n_grid):
            i = idx(r, c)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if criterio == "rook" and abs(dr) + abs(dc) != 1:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n_grid and 0 <= cc < n_grid:
                        W[i, idx(rr, cc)] = 1.0
    if estandarizar:
        sumas = W.sum(axis=1, keepdims=True)
        sumas[sumas == 0] = 1.0
        W = W / sumas
    return W


@dataclass
class ResultadoMoran:
    """Resultado del Índice de Moran global."""

    I: float
    esperado: float
    p_valor: float
    z_score: float
    n_perm: int

    def interpreta(self) -> str:
        if self.p_valor >= 0.05:
            return "sin dependencia espacial significativa (p >= 0.05)"
        if self.I > self.esperado:
            return "autocorrelación positiva: valores similares se agrupan"
        return "autocorrelación negativa: valores contrastan con sus vecinos"


def moran_global(
    valores: np.ndarray,
    W: np.ndarray,
    n_perm: int = 999,
    seed: int = 42,
) -> ResultadoMoran:
    """Índice de Moran global con p-valor por permutaciones.

    Parameters
    ----------
    valores:
        Vector plano de la variable por celda (orden fila-mayor, longitud
        ``n_grid**2``).
    W:
        Matriz de pesos espaciales.
    """
    x = np.asarray(valores, dtype=float)
    n = len(x)
    z = x - x.mean()
    s0 = W.sum()
    denom = np.sum(z * z)
    if denom == 0 or s0 == 0:
        return ResultadoMoran(np.nan, -1 / (n - 1), np.nan, np.nan, n_perm)

    def calc_I(zv):
        return (n / s0) * (zv @ (W @ zv)) / np.sum(zv * zv)

    I_obs = calc_I(z)
    esperado = -1.0 / (n - 1)

    rng = np.random.default_rng(seed)
    perms = np.empty(n_perm)
    for k in range(n_perm):
        zp = rng.permutation(z)
        perms[k] = calc_I(zp)

    # p-valor de dos colas basado en permutaciones.
    mayor = np.sum(np.abs(perms - esperado) >= np.abs(I_obs - esperado))
    p_val = (mayor + 1) / (n_perm + 1)
    z_score = (I_obs - perms.mean()) / (perms.std() + 1e-12)
    return ResultadoMoran(float(I_obs), float(esperado), float(p_val),
                          float(z_score), n_perm)


@dataclass
class ResultadoLISA:
    """Resultado del Índice de Moran local (LISA)."""

    I_local: np.ndarray       # I local por celda
    p_valor: np.ndarray       # p-valor por celda (permutaciones)
    cluster: np.ndarray       # etiqueta: HH, LL, HL, LH, ns


def moran_local(
    valores: np.ndarray,
    W: np.ndarray,
    n_perm: int = 999,
    seed: int = 42,
    alpha: float = 0.05,
) -> ResultadoLISA:
    """Índice de Moran local (LISA) para detectar clústeres espaciales.

    Etiqueta cada celda como HH (alta rodeada de alta), LL, HL, LH o ``ns``
    (no significativa) según su valor y el de su entorno.
    """
    x = np.asarray(valores, dtype=float)
    n = len(x)
    z = x - x.mean()
    m2 = np.sum(z * z) / n
    if m2 == 0:
        return ResultadoLISA(
            np.zeros(n), np.ones(n), np.array(["ns"] * n)
        )

    lag = W @ z
    I_local = (z / m2) * lag

    rng = np.random.default_rng(seed)
    p_val = np.ones(n)
    # Permutación condicional vectorizada: para cada celda i se remuestrean sus
    # vecinos entre las demás celdas (los pesos row-standarizados actúan como
    # promedio ponderado sobre los valores muestreados).
    for i in range(n):
        w_i = W[i]
        idx_vec = np.nonzero(w_i)[0]
        q = len(idx_vec)
        if q == 0:
            continue
        pesos = w_i[idx_vec]
        otros = np.delete(z, i)
        # Muestrear q valores por permutación (sin reemplazo) para n_perm réplicas.
        muestras = np.array([
            rng.choice(otros, size=q, replace=False) for _ in range(n_perm)
        ])
        lag_sims = muestras @ pesos
        sims = (z[i] / m2) * lag_sims
        mayor = np.sum(np.abs(sims) >= np.abs(I_local[i]))
        p_val[i] = (mayor + 1) / (n_perm + 1)

    zlag = lag
    cluster = np.array(["ns"] * n, dtype=object)
    sig = p_val < alpha
    cluster[sig & (z > 0) & (zlag > 0)] = "HH"
    cluster[sig & (z < 0) & (zlag < 0)] = "LL"
    cluster[sig & (z > 0) & (zlag < 0)] = "HL"
    cluster[sig & (z < 0) & (zlag > 0)] = "LH"
    return ResultadoLISA(I_local, p_val, cluster)


def variable_coocurrencia(df_celdas: np.ndarray) -> np.ndarray:
    """Codifica el estado de coocurrencia por celda para el análisis espacial.

    Convención: 2 = ambos (verde), 1 = un solo evento (azul/amarillo),
    0 = blanca. Útil como variable ordinal para Moran.
    """
    return np.asarray(df_celdas, dtype=float)

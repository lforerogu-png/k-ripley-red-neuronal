r"""K de Ripley univariada y cruzada con corrección de borde isotrópica.

Se implementa **directamente** la fórmula del estimador de Ripley con la
corrección isotrópica (equivalente a ``spatstat::Kest(correction="iso")`` y
``spatstat::Kcross(correction="iso")``), en lugar de depender de un paquete
externo. Esto permite controlar exactamente el cálculo y validarlo.

Estimador univariado (corrección isotrópica)
--------------------------------------------
.. math::

    \hat{K}(r) = \frac{|A|}{n(n-1)}
        \sum_{i \ne j} \mathbf{1}(d_{ij} \le r)\, e_{ij},

donde :math:`|A|` es el área de la ventana, :math:`d_{ij}` la distancia
euclidiana y :math:`e_{ij} = 1 / w(i, d_{ij})` el peso de corrección
isotrópico: :math:`w(i, d)` es la proporción de la circunferencia de radio
:math:`d` centrada en el punto :math:`i` que cae **dentro** de la ventana
rectangular. Para el proceso de Poisson homogéneo (CSR): :math:`K(r)=\pi r^2`.

Estimador cruzado (bivariado)
-----------------------------
.. math::

    \hat{K}_{AB}(r) = \frac{|A|}{n_A n_B}
        \sum_{i \in A}\sum_{j \in B} \mathbf{1}(d_{ij} \le r)\, e_{ij}.

Clasificación
-------------
Se ofrecen **dos** métodos, tal como pide el enunciado:

1. Umbral fijo respecto a :math:`\pi r^2` (como en el R original), con banda
   de tolerancia configurable.
2. Envolventes de simulación Monte Carlo bajo CSR / independencia (las
   "bandas envolventes" que el profesor describe en el audio): si la curva
   cae dentro del envolvente -> aleatorio/independiente; por encima ->
   agrupado/atracción; por debajo -> disperso/repulsión.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Corrección de borde isotrópica
# ---------------------------------------------------------------------------
def fraccion_circulo_dentro(
    x: np.ndarray,
    y: np.ndarray,
    radio: np.ndarray,
    ventana: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
) -> np.ndarray:
    r"""Proporción de la circunferencia dentro de una ventana rectangular.

    Para un círculo de radio ``radio`` centrado en ``(x, y)`` dentro del
    rectángulo ``ventana = (xmin, xmax, ymin, ymax)``, devuelve la fracción
    del perímetro que queda **dentro** del rectángulo. Es el término
    :math:`w(i, d)` de la corrección isotrópica.

    Deducción: un punto de la circunferencia en ángulo :math:`\varphi` está
    fuera por el borde izquierdo si :math:`\cos\varphi < -b/ r` con
    :math:`b` la distancia al borde. Esto define un arco "fuera" centrado en
    la normal exterior del borde y de semiancho :math:`\alpha=\arccos(b/r)`.
    Para :math:`r < 0.5` en el cuadrado unitario solo pueden solaparse arcos
    de bordes **adyacentes** (esquinas); ese solape se resta con
    inclusión-exclusión exacta.

    Todos los argumentos se difunden (broadcasting) elemento a elemento.
    """
    xmin, xmax, ymin, ymax = ventana
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.asarray(radio, dtype=float)
    r_safe = np.where(r <= 0, np.nan, r)

    # Distancias a cada borde y semiángulo del arco exterior correspondiente.
    def semiangulo(b):
        ratio = np.clip(b / r_safe, -1.0, 1.0)
        alpha = np.arccos(ratio)          # en (0, pi/2] cuando b < r
        return np.where(b < r_safe, alpha, 0.0)

    aL = semiangulo(x - xmin)   # borde izquierdo  (normal exterior en pi)
    aR = semiangulo(xmax - x)   # borde derecho    (normal exterior en 0)
    aB = semiangulo(y - ymin)   # borde inferior   (normal exterior en 3pi/2)
    aT = semiangulo(ymax - y)   # borde superior   (normal exterior en pi/2)

    # Longitud total de arcos exteriores (cada arco mide 2*alpha).
    total = 2.0 * (aL + aR + aB + aT)

    # Solapes en las cuatro esquinas (pares de bordes adyacentes).
    media_pi = np.pi / 2.0
    solape = (
        np.maximum(0.0, aR + aT - media_pi)
        + np.maximum(0.0, aT + aL - media_pi)
        + np.maximum(0.0, aL + aB - media_pi)
        + np.maximum(0.0, aB + aR - media_pi)
    )
    fuera = total - solape
    dentro = 1.0 - fuera / (2.0 * np.pi)
    return np.clip(dentro, 1e-6, 1.0)


def _matriz_distancias(pts: np.ndarray) -> np.ndarray:
    """Matriz de distancias euclidianas ``n x n``."""
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _distancias_cruzadas(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matriz de distancias ``n_a x n_b`` entre dos conjuntos de puntos."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _acumular_por_radio(
    dist: np.ndarray, pesos: np.ndarray, radios: np.ndarray
) -> np.ndarray:
    """Suma de ``pesos`` de pares con ``dist <= r`` para cada ``r`` en ``radios``.

    Implementación O(m log m) mediante ordenación de distancias y suma
    acumulada, mucho más rápida que enmascarar por cada radio (crucial para
    calcular envolventes de Monte Carlo).
    """
    orden = np.argsort(dist, kind="quicksort")
    d_ord = dist[orden]
    w_ord = pesos[orden]
    cum = np.cumsum(w_ord)
    idx = np.searchsorted(d_ord, radios, side="right")
    resultado = np.zeros_like(radios, dtype=float)
    mask = idx > 0
    resultado[mask] = cum[idx[mask] - 1]
    return resultado


@dataclass
class ResultadoK:
    """Curva K de Ripley evaluada en una malla de radios ``r``."""

    r: np.ndarray
    k: np.ndarray
    k_teorica: np.ndarray  # pi * r^2

    def valor_en(self, r_ref: float) -> float:
        """Devuelve K interpolada/al radio más cercano a ``r_ref``."""
        if len(self.r) == 0:
            return float(np.pi * r_ref**2)
        idx = int(np.argmin(np.abs(self.r - r_ref)))
        val = self.k[idx]
        return float(val) if np.isfinite(val) else float(np.pi * r_ref**2)


@dataclass
class ResultadoL:
    """Curva L de Ripley (transformación de Besag) evaluada en una malla ``r``.

    Se define como :math:`L(r) = \\sqrt{K(r)/\\pi} - r`. Bajo CSR,
    :math:`K(r)=\\pi r^2` y por tanto :math:`L(r)=0`.
    """

    r: np.ndarray
    l: np.ndarray
    l_teorica: np.ndarray  # 0 bajo CSR

    def valor_en(self, r_ref: float) -> float:
        """Devuelve L interpolada/al radio más cercano a ``r_ref``."""
        if len(self.r) == 0:
            return 0.0
        idx = int(np.argmin(np.abs(self.r - r_ref)))
        val = self.l[idx]
        return float(val) if np.isfinite(val) else 0.0


def k_a_l(res: ResultadoK) -> ResultadoL:
    """Transforma una curva K en la función L de Ripley (Besag, 1977).

    .. math::
        L(r) = \\sqrt{K(r)/\\pi} - r

    Bajo CSR (:math:`K(r)=\\pi r^2`) resulta :math:`L(r)=0`. Valores
    positivos indican agrupamiento/atracción; negativos, dispersión/repulsión.
    """
    k_segura = np.maximum(np.asarray(res.k, dtype=float), 0.0)
    l = np.sqrt(k_segura / np.pi) - res.r
    l_teorica = np.zeros_like(res.r, dtype=float)
    return ResultadoL(r=res.r, l=l, l_teorica=l_teorica)


def rmax_por_defecto(area: float = 1.0) -> float:
    """Radio máximo por defecto (regla de spatstat: 1/4 del lado)."""
    return 0.25 * np.sqrt(area)


def _malla_radios(rmax: float, n: int = 128) -> np.ndarray:
    return np.linspace(0.0, rmax, n)


def k_univariada(
    puntos_xy: np.ndarray,
    ventana: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    rmax: Optional[float] = None,
    n_r: int = 128,
) -> ResultadoK:
    """Estima la K de Ripley univariada con corrección isotrópica.

    Parameters
    ----------
    puntos_xy:
        Array ``(n, 2)`` de coordenadas en la ventana.
    ventana:
        Rectángulo ``(xmin, xmax, ymin, ymax)``.
    rmax:
        Radio máximo; por defecto 1/4 del lado.
    n_r:
        Número de radios en la malla.
    """
    xmin, xmax, ymin, ymax = ventana
    area = (xmax - xmin) * (ymax - ymin)
    if rmax is None:
        rmax = rmax_por_defecto(area)
    r = _malla_radios(rmax, n_r)

    pts = np.asarray(puntos_xy, dtype=float)
    n = len(pts)
    if n < 2:
        return ResultadoK(r=r, k=np.full_like(r, np.nan), k_teorica=np.pi * r**2)

    d = _matriz_distancias(pts)
    # Pesos de corrección: círculo centrado en i de radio d_ij.
    xi = np.repeat(pts[:, 0][:, None], n, axis=1)
    yi = np.repeat(pts[:, 1][:, None], n, axis=1)
    w = fraccion_circulo_dentro(xi, yi, d, ventana)
    e = 1.0 / w
    np.fill_diagonal(d, np.inf)   # excluir i == j (no contribuye: d = inf)

    factor = area / (n * (n - 1))
    k = _acumular_por_radio(d.ravel(), e.ravel(), r) * factor
    return ResultadoK(r=r, k=k, k_teorica=np.pi * r**2)


# Direcciones estándar para el análisis de anisotropía (grados desde el eje x).
DIRECCIONES_ANISOTROPIA: tuple[int, ...] = (0, 45, 90, 135)
ETIQUETAS_DIRECCION: dict[int, str] = {
    0: "0° (horizontal)",
    45: "45° (diagonal)",
    90: "90° (vertical)",
    135: "135° (diagonal inversa)",
}


def _en_sector_angular(
    angulos: np.ndarray, centro_grados: float, tolerancia_grados: float
) -> np.ndarray:
    """Máscara booleana: ángulos (rad) dentro del sector angular."""
    centro = np.deg2rad(centro_grados)
    tol = np.deg2rad(tolerancia_grados)
    diff = np.angle(np.exp(1j * (angulos - centro)))
    return np.abs(diff) <= tol


def k_direccional(
    puntos_xy: np.ndarray,
    angulo_grados: float,
    tolerancia_grados: float = 22.5,
    ventana: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    rmax: Optional[float] = None,
    n_r: int = 128,
) -> ResultadoK:
    """K de Ripley univariada restringida a un sector angular.

    Solo cuenta pares cuya dirección (de :math:`i` hacia :math:`j`) cae
    dentro de ``[angulo_grados ± tolerancia_grados]``. El estimador se
    escala por el ancho del sector para que, bajo CSR isotrópico, cada
    curva direccional coincida con la referencia :math:`\\pi r^2`.
    """
    xmin, xmax, ymin, ymax = ventana
    area = (xmax - xmin) * (ymax - ymin)
    if rmax is None:
        rmax = rmax_por_defecto(area)
    r = _malla_radios(rmax, n_r)

    pts = np.asarray(puntos_xy, dtype=float)
    n = len(pts)
    if n < 2:
        return ResultadoK(r=r, k=np.full_like(r, np.nan), k_teorica=np.pi * r**2)

    d = _matriz_distancias(pts)
    xi = np.repeat(pts[:, 0][:, None], n, axis=1)
    yi = np.repeat(pts[:, 1][:, None], n, axis=1)
    w = fraccion_circulo_dentro(xi, yi, d, ventana)
    e = 1.0 / w

    dx = pts[None, :, 0] - pts[:, 0][:, None]
    dy = pts[None, :, 1] - pts[:, 1][:, None]
    angulos = np.arctan2(dy, dx)
    sector = _en_sector_angular(angulos, angulo_grados, tolerancia_grados)
    np.fill_diagonal(sector, False)
    np.fill_diagonal(d, np.inf)

    e_dir = np.where(sector, e, 0.0)
    ancho_sector = 2.0 * tolerancia_grados / 360.0
    factor = area / (n * (n - 1)) / ancho_sector
    k = _acumular_por_radio(d.ravel(), e_dir.ravel(), r) * factor
    return ResultadoK(r=r, k=k, k_teorica=np.pi * r**2)


def k_anisotropia(
    puntos_xy: np.ndarray,
    direcciones: tuple[int, ...] = DIRECCIONES_ANISOTROPIA,
    tolerancia_grados: float = 22.5,
    ventana: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    rmax: Optional[float] = None,
    n_r: int = 128,
) -> dict[int, ResultadoK]:
    """Calcula K direccional en varias orientaciones para análisis de anisotropía."""
    return {
        ang: k_direccional(
            puntos_xy, ang, tolerancia_grados,
            ventana=ventana, rmax=rmax, n_r=n_r,
        )
        for ang in direcciones
    }


def clasificar_anisotropia(
    curvas: dict[int, ResultadoK],
    r_ref: float = 0.2,
    umbral: float = 0.20,
) -> tuple[str, float]:
    """Clasifica isotropía comparando las curvas direccionales en ``r_ref``.

    Si la diferencia relativa ``(max - min) / (π r_ref²)`` supera ``umbral``
    (20 % por defecto), el patrón se considera anisotrópico.
    """
    if not curvas:
        return "Patrón isotrópico", 0.0
    valores = [res.valor_en(r_ref) for res in curvas.values()]
    k_teo = np.pi * r_ref**2
    if k_teo <= 0:
        return "Patrón isotrópico", 0.0
    diff_rel = (max(valores) - min(valores)) / k_teo
    etiqueta = "Patrón anisotrópico" if diff_rel > umbral else "Patrón isotrópico"
    return etiqueta, float(diff_rel)


def k_cruzada(
    a_xy: np.ndarray,
    b_xy: np.ndarray,
    ventana: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    rmax: Optional[float] = None,
    n_r: int = 128,
) -> ResultadoK:
    """Estima la K de Ripley cruzada (bivariada) A->B con corrección iso.

    Parameters
    ----------
    a_xy, b_xy:
        Arrays ``(n_a, 2)`` y ``(n_b, 2)`` de los dos tipos de puntos.
    """
    xmin, xmax, ymin, ymax = ventana
    area = (xmax - xmin) * (ymax - ymin)
    if rmax is None:
        rmax = rmax_por_defecto(area)
    r = _malla_radios(rmax, n_r)

    a = np.asarray(a_xy, dtype=float)
    b = np.asarray(b_xy, dtype=float)
    na, nb = len(a), len(b)
    if na < 1 or nb < 1:
        return ResultadoK(r=r, k=np.full_like(r, np.nan), k_teorica=np.pi * r**2)

    d = _distancias_cruzadas(a, b)               # (na, nb)
    xi = np.repeat(a[:, 0][:, None], nb, axis=1)
    yi = np.repeat(a[:, 1][:, None], nb, axis=1)
    w = fraccion_circulo_dentro(xi, yi, d, ventana)
    e = 1.0 / w

    factor = area / (na * nb)
    k = _acumular_por_radio(d.ravel(), e.ravel(), r) * factor
    return ResultadoK(r=r, k=k, k_teorica=np.pi * r**2)


# ---------------------------------------------------------------------------
# Clasificación por umbral fijo (como en el R original)
# ---------------------------------------------------------------------------
def clasificar_univariada_umbral(
    res: ResultadoK, r_ref: float = 0.2, tol: float = 0.15
) -> str:
    """Clasifica un patrón por umbral fijo respecto a ``pi*r^2``.

    Devuelve ``"agrupado"`` si K > (1+tol)*K_teo, ``"disperso"`` si
    K < (1-tol)*K_teo, y ``"aleatorio"`` en caso intermedio.
    """
    k_teo = np.pi * r_ref**2
    val = res.valor_en(r_ref)
    if val > k_teo * (1 + tol):
        return "agrupado"
    if val < k_teo * (1 - tol):
        return "disperso"
    return "aleatorio"


def clasificar_bivariada_umbral(
    res: ResultadoK, r_ref: float = 0.2, tol: float = 0.15
) -> str:
    """Clasifica la coocurrencia por umbral fijo respecto a ``pi*r^2``."""
    k_teo = np.pi * r_ref**2
    val = res.valor_en(r_ref)
    if val > k_teo * (1 + tol):
        return "atraccion"
    if val < k_teo * (1 - tol):
        return "repulsion"
    return "sin_coocurrencia"


# ---------------------------------------------------------------------------
# Clasificación por envolventes de simulación (Monte Carlo)
# ---------------------------------------------------------------------------
@dataclass
class Envolvente:
    """Banda envolvente de simulación de K bajo la hipótesis nula."""

    r: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    media: np.ndarray


def envolvente_k_a_l(env: Envolvente) -> Envolvente:
    """Convierte una envolvente de K a la escala L."""
    def _conv(curva: np.ndarray) -> np.ndarray:
        k_segura = np.maximum(np.asarray(curva, dtype=float), 0.0)
        return np.sqrt(k_segura / np.pi) - env.r

    return Envolvente(
        r=env.r,
        lo=_conv(env.lo),
        hi=_conv(env.hi),
        media=_conv(env.media),
    )


def envolvente_csr(
    n_puntos: int,
    ventana: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    n_sim: int = 99,
    rmax: Optional[float] = None,
    n_r: int = 128,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Envolvente:
    """Envolvente de K univariada bajo CSR (Poisson homogéneo).

    Simula ``n_sim`` patrones de ``n_puntos`` puntos uniformes y toma los
    cuantiles ``alpha/2`` y ``1-alpha/2`` de las curvas K por radio.
    """
    xmin, xmax, ymin, ymax = ventana
    rng = np.random.default_rng(seed)
    if rmax is None:
        rmax = rmax_por_defecto((xmax - xmin) * (ymax - ymin))
    curvas = np.empty((n_sim, n_r))
    for s in range(n_sim):
        xs = rng.uniform(xmin, xmax, n_puntos)
        ys = rng.uniform(ymin, ymax, n_puntos)
        res = k_univariada(np.column_stack([xs, ys]), ventana, rmax, n_r)
        curvas[s] = res.k
    r = _malla_radios(rmax, n_r)
    lo = np.nanquantile(curvas, alpha / 2, axis=0)
    hi = np.nanquantile(curvas, 1 - alpha / 2, axis=0)
    media = np.nanmean(curvas, axis=0)
    return Envolvente(r=r, lo=lo, hi=hi, media=media)


def envolvente_independencia(
    a_xy: np.ndarray,
    b_xy: np.ndarray,
    ventana: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    n_sim: int = 99,
    rmax: Optional[float] = None,
    n_r: int = 128,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Envolvente:
    """Envolvente de K cruzada bajo independencia (re-etiquetado aleatorio).

    Mantiene las localizaciones combinadas y reasigna aleatoriamente las
    etiquetas A/B preservando ``n_a`` y ``n_b``. Es el nulo de "mezcla
    aleatoria" apropiado para K cruzada.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a_xy, float)
    b = np.asarray(b_xy, float)
    todos = np.vstack([a, b]) if len(a) and len(b) else np.vstack(
        [x for x in (a, b) if len(x)]
    )
    na, nb = len(a), len(b)
    if rmax is None:
        xmin, xmax, ymin, ymax = ventana
        rmax = rmax_por_defecto((xmax - xmin) * (ymax - ymin))
    curvas = np.empty((n_sim, n_r))
    n_total = len(todos)
    for s in range(n_sim):
        perm = rng.permutation(n_total)
        idx_a = perm[:na]
        idx_b = perm[na:na + nb]
        res = k_cruzada(todos[idx_a], todos[idx_b], ventana, rmax, n_r)
        curvas[s] = res.k
    r = _malla_radios(rmax, n_r)
    lo = np.nanquantile(curvas, alpha / 2, axis=0)
    hi = np.nanquantile(curvas, 1 - alpha / 2, axis=0)
    media = np.nanmean(curvas, axis=0)
    return Envolvente(r=r, lo=lo, hi=hi, media=media)


def clasificar_por_envolvente(
    res: ResultadoK,
    env: Envolvente,
    r_ref: float = 0.2,
    modo: str = "univariada",
) -> str:
    """Clasifica comparando K observada con la banda envolvente en ``r_ref``.

    Parameters
    ----------
    modo:
        ``"univariada"`` -> agrupado/aleatorio/disperso;
        ``"bivariada"`` -> atraccion/sin_coocurrencia/repulsion.
    """
    idx = int(np.argmin(np.abs(env.r - r_ref)))
    val = res.valor_en(r_ref)
    lo, hi = env.lo[idx], env.hi[idx]
    if modo == "univariada":
        if val > hi:
            return "agrupado"
        if val < lo:
            return "disperso"
        return "aleatorio"
    else:
        if val > hi:
            return "atraccion"
        if val < lo:
            return "repulsion"
        return "sin_coocurrencia"

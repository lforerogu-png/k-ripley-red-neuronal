"""Utilidades de visualización (matplotlib) — tema académico oscuro.

Funciones para dibujar patrones, grillas con los 4 colores
(blanco/azul/amarillo/verde), curvas de K con envolventes, matrices de
confusión, curvas de aprendizaje y mapas LISA. Cada función devuelve una
figura de matplotlib para poder guardarla o incrustarla en Streamlit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

from . import COLOR_AMBOS, COLOR_BLANCA, COLOR_SOLO_A, COLOR_SOLO_B

# Paleta académica oscura (alineada con la app Streamlit)
BG_MAIN = "#0f1117"
BG_PANEL = "#1a1d27"
BG_FORMULA = "#1e2d4a"
TEXT_PRIMARY = "#dde1f0"
TEXT_SECONDARY = "#7b82a0"
BORDER = "#252836"
ACCENT = "#4a9eff"
ACCENT_LIGHT = "#64b5f6"
CSR_GRAY = "#4a4d6a"
GRID_LINE = "#3a3d52"
POSITIVE = "#66bb6a"
WARNING = "#ffa726"
NEGATIVE = "#ef5350"

_MAPA_CLS = {
    "agrupado": "Agrupado",
    "disperso": "Disperso",
    "aleatorio": "Aleatorio",
    "atraccion": "Atracción",
    "repulsion": "Repulsión",
    "sin_coocurrencia": "Independencia",
}


def _aplicar_estilo():
    """Configura rcParams globales para figuras con tema oscuro."""
    plt.rcParams.update({
        "figure.facecolor": BG_MAIN,
        "axes.facecolor": BG_PANEL,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "600",
        "axes.grid": False,
        "legend.facecolor": (0.1, 0.11, 0.15, 0.85),
        "legend.edgecolor": "none",
        "legend.labelcolor": TEXT_PRIMARY,
        "figure.autolayout": False,
    })


def _estilo_ax(ax, *, titulo: Optional[str] = None):
    """Aplica estilo oscuro a un eje matplotlib."""
    ax.set_facecolor(BG_PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    if titulo:
        ax.set_title(titulo, fontsize=10, fontweight="600", color=TEXT_PRIMARY, pad=8)


def _etiqueta_clasificacion(clase: Optional[str]) -> str:
    if not clase:
        return ""
    return _MAPA_CLS.get(str(clase).lower(), str(clase).capitalize())


def _cmap_confusion():
    return LinearSegmentedColormap.from_list("academic_blue", [BG_PANEL, ACCENT])


_aplicar_estilo()


def matriz_patron(p, n_grid: int = 30) -> np.ndarray:
    """Matriz binaria 0/1: 1 si la celda está ocupada por el patrón."""
    mat = np.zeros((n_grid, n_grid), dtype=int)
    if len(p):
        for c, r in zip(p["col"], p["row"]):
            if 1 <= c <= n_grid and 1 <= r <= n_grid:
                mat[int(r) - 1, int(c) - 1] = 1
    return mat


def _dibujar_grilla_celdas(ax, n_grid: int, color: str = GRID_LINE, lw: float = 0.25):
    """Líneas de separación entre celdas."""
    for i in range(n_grid + 1):
        y = 0.5 + i
        ax.axhline(y, color=color, linewidth=lw, zorder=2)
        ax.axvline(y, color=color, linewidth=lw, zorder=2)


def fig_patron(p, color, titulo, n_grid=30):
    """Dibuja un patrón como cuadrados rellenos sobre grilla N×N."""
    ocupacion = matriz_patron(p, n_grid)
    cmap = ListedColormap([BG_PANEL, color])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor=BG_MAIN)
    ax.imshow(
        ocupacion,
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=(0.5, n_grid + 0.5, 0.5, n_grid + 0.5),
        interpolation="nearest",
        zorder=1,
    )
    _dibujar_grilla_celdas(ax, n_grid)
    ax.set_xlim(0.5, n_grid + 0.5)
    ax.set_ylim(0.5, n_grid + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    _estilo_ax(ax, titulo=titulo)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    return fig


def matriz_estados(p1, p2, n_grid=30) -> np.ndarray:
    """Codifica la grilla: 0=blanca, 1=solo A, 2=solo B, 3=ambos."""
    estado = np.zeros((n_grid, n_grid), dtype=int)
    s1 = set(zip(p1["col"], p1["row"])) if len(p1) else set()
    s2 = set(zip(p2["col"], p2["row"])) if len(p2) else set()
    for c, r in s1:
        estado[r - 1, c - 1] += 1
    for c, r in s2:
        estado[r - 1, c - 1] += 2
    return estado


def fig_grilla_colores(
    p1,
    p2,
    n_grid=30,
    titulo="Superposición",
    resultado: Optional[str] = None,
):
    """Grilla con los 4 colores: fondo oscuro, cuadrícula y anotación de resultado."""
    estado = matriz_estados(p1, p2, n_grid)
    cmap = ListedColormap([BG_PANEL, COLOR_SOLO_A, COLOR_SOLO_B, COLOR_AMBOS])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor=BG_MAIN)
    ax.imshow(
        estado,
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=(0.5, n_grid + 0.5, 0.5, n_grid + 0.5),
        interpolation="nearest",
        zorder=1,
    )
    _dibujar_grilla_celdas(ax, n_grid)
    ax.set_xlim(0.5, n_grid + 0.5)
    ax.set_ylim(0.5, n_grid + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    _estilo_ax(ax, titulo=titulo)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if resultado:
        etiqueta = _etiqueta_clasificacion(resultado)
        ax.text(
            0.02, 0.98, f"Resultado: {etiqueta}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=TEXT_SECONDARY,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_PANEL,
                      edgecolor=BORDER, alpha=0.9),
        )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    return fig


def fig_curva_k(
    res,
    titulo,
    color=ACCENT,
    envolvente=None,
    r_ref=0.2,
    clasificacion: Optional[str] = None,
):
    """Curva K observada vs CSR con relleno, anotación y tema oscuro."""
    fig, ax = plt.subplots(figsize=(3.8, 3.2), facecolor=BG_MAIN)
    _estilo_ax(ax, titulo=titulo)

    ax.fill_between(res.r, res.k, res.k_teorica, color=color, alpha=0.10, zorder=1)
    ax.plot(res.r, res.k, color=color, lw=2, label="K observada", zorder=3)
    ax.plot(res.r, res.k_teorica, "--", color=CSR_GRAY, lw=1.2, label="CSR", zorder=2)
    if envolvente is not None:
        ax.fill_between(
            envolvente.r, envolvente.lo, envolvente.hi,
            color=CSR_GRAY, alpha=0.18, label="Envolvente CSR", zorder=1,
        )
    ax.axvline(r_ref, color=CSR_GRAY, ls=":", lw=1, alpha=0.7, zorder=2)

    if clasificacion:
        etiqueta = _etiqueta_clasificacion(clasificacion)
        ax.text(
            0.98, 0.98, etiqueta,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=TEXT_PRIMARY, fontweight="500",
        )

    ax.set_xlabel("Radio r")
    ax.set_ylabel("K(r)")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.18)
    return fig


def fig_curva_l(
    res,
    titulo,
    color=ACCENT,
    envolvente=None,
    r_ref=0.2,
    clasificacion: Optional[str] = None,
):
    """Curva L observada vs CSR (L=0) con relleno, anotación y tema oscuro."""
    fig, ax = plt.subplots(figsize=(3.8, 3.2), facecolor=BG_MAIN)
    _estilo_ax(ax, titulo=titulo)

    ax.fill_between(res.r, res.l, res.l_teorica, color=color, alpha=0.10, zorder=1)
    ax.plot(res.r, res.l, color=color, lw=2, label="L observada", zorder=3)
    ax.plot(res.r, res.l_teorica, "--", color=CSR_GRAY, lw=1.2, label="CSR (L=0)", zorder=2)
    if envolvente is not None:
        ax.fill_between(
            envolvente.r, envolvente.lo, envolvente.hi,
            color=CSR_GRAY, alpha=0.18, label="Envolvente CSR", zorder=1,
        )
    ax.axvline(r_ref, color=CSR_GRAY, ls=":", lw=1, alpha=0.7, zorder=2)
    ax.axhline(0.0, color=CSR_GRAY, ls="-", lw=0.6, alpha=0.35, zorder=1)

    if clasificacion:
        etiqueta = _etiqueta_clasificacion(clasificacion)
        ax.text(
            0.98, 0.98, etiqueta,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=TEXT_PRIMARY, fontweight="500",
        )

    ax.set_xlabel("Radio r")
    ax.set_ylabel("L(r)")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.18)
    return fig


_COLORES_DIRECCION = {
    0: ACCENT,
    45: WARNING,
    90: POSITIVE,
    135: "#ce93d8",
}


def fig_k_direccional(
    curvas: dict,
    titulo: str,
    etiquetas: Optional[dict] = None,
    r_ref: float = 0.2,
):
    """Varias curvas K direccionales con referencia CSR en un solo panel."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor=BG_MAIN)
    _estilo_ax(ax, titulo=titulo)

    primera = next(iter(curvas.values()))
    ax.plot(
        primera.r, primera.k_teorica, "--",
        color=CSR_GRAY, lw=1.2, label="CSR (πr²)", zorder=2,
    )
    for angulo, res in curvas.items():
        color = _COLORES_DIRECCION.get(angulo, ACCENT_LIGHT)
        if etiquetas and angulo in etiquetas:
            etiqueta = etiquetas[angulo]
        else:
            etiqueta = f"{angulo}°"
        ax.plot(res.r, res.k, color=color, lw=1.8, label=etiqueta, zorder=3)

    ax.axvline(r_ref, color=CSR_GRAY, ls=":", lw=1, alpha=0.7, zorder=2)
    ax.set_xlabel("Radio r")
    ax.set_ylabel("K(r)")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.18)
    return fig


def fig_curvas_aprendizaje(resultado, titulo="Curvas de aprendizaje"):
    """Curvas de pérdida train/val con sombreado de sobreajuste."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2), facecolor=BG_MAIN)
    _estilo_ax(ax, titulo=titulo)

    epocas = range(1, len(resultado.loss_train) + 1)
    ax.plot(epocas, resultado.loss_train, label="Pérdida train", color=ACCENT, lw=1.8)

    if resultado.loss_val:
        ax.plot(epocas, resultado.loss_val, label="Pérdida val", color=NEGATIVE, lw=1.8)
        train_arr = np.asarray(resultado.loss_train)
        val_arr = np.asarray(resultado.loss_val)
        sobreajuste = val_arr > train_arr
        if np.any(sobreajuste):
            ax.fill_between(
                epocas, train_arr, val_arr,
                where=sobreajuste, color=NEGATIVE, alpha=0.05,
                interpolate=True, label="_nolegend_",
            )
        min_val = float(np.nanmin(val_arr))
        min_ep = int(np.nanargmin(val_arr)) + 1
        ax.axhline(min_val, color=TEXT_SECONDARY, ls="--", lw=0.8, alpha=0.6)
        ax.text(
            len(epocas) * 0.98, min_val, " mínimo val.",
            ha="right", va="bottom", fontsize=7, color=TEXT_SECONDARY,
        )

    ax.set_xlabel("Época")
    ax.set_ylabel("Log-loss")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.18)
    return fig


def fig_matriz_confusion(cm: pd.DataFrame, titulo="Matriz de confusión"):
    """Heatmap académico con conteos absolutos y porcentajes por fila."""
    fig, ax = plt.subplots(figsize=(4.2, 3.6), facecolor=BG_MAIN)
    _estilo_ax(ax, titulo=titulo)

    data = cm.to_numpy().astype(float)
    totales_fila = data.sum(axis=1, keepdims=True)
    totales_fila = np.where(totales_fila == 0, 1, totales_fila)
    pct = data / totales_fila * 100

    cmap = _cmap_confusion()
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=max(data.max(), 1))
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(
        [c.replace("pred_", "") for c in cm.columns],
        rotation=30, ha="right", fontsize=8, color=TEXT_SECONDARY,
    )
    ax.set_yticklabels(
        [i.replace("real_", "") for i in cm.index],
        fontsize=8, color=TEXT_SECONDARY,
    )
    ax.set_xlabel("Predicho", color=TEXT_SECONDARY)
    ax.set_ylabel("Real", color=TEXT_SECONDARY)

    umbral = data.max() / 2 if data.max() else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = int(data[i, j])
            p = pct[i, j]
            ax.text(
                j, i - 0.12, str(val), ha="center", va="center",
                color=TEXT_PRIMARY if data[i, j] > umbral else TEXT_SECONDARY,
                fontsize=10, fontweight="600",
            )
            ax.text(
                j, i + 0.18, f"{p:.0f}%", ha="center", va="center",
                color=TEXT_SECONDARY, fontsize=7,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY)
    cbar.outline.set_edgecolor(BORDER)
    fig.subplots_adjust(left=0.18, right=0.88, top=0.88, bottom=0.22)
    return fig


def fig_lisa(cluster: np.ndarray, n_grid: int, titulo="Clústeres LISA"):
    """Mapa de clústeres LISA sobre la grilla."""
    codigo = {"ns": 0, "HH": 1, "LL": 2, "HL": 3, "LH": 4}
    colores = [BG_PANEL, NEGATIVE, ACCENT, WARNING, "#9c7bd8"]
    mat = np.array([codigo.get(c, 0) for c in cluster]).reshape(n_grid, n_grid)
    cmap = ListedColormap(colores)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor=BG_MAIN)
    ax.imshow(mat, cmap=cmap, norm=norm, origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    _estilo_ax(ax, titulo=titulo)
    from matplotlib.patches import Patch
    leyenda = [Patch(facecolor=colores[v], edgecolor=BORDER, label=k)
               for k, v in codigo.items()]
    ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=5, fontsize=7, frameon=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.12)
    return fig


def fig_dispersion_2d(df, titulo="Proyección 2D"):
    """Scatter 2D coloreado por clase (para PCA/t-SNE)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.8), facecolor=BG_MAIN)
    _estilo_ax(ax, titulo=titulo)
    palette = [ACCENT, WARNING, POSITIVE, NEGATIVE, ACCENT_LIGHT]
    for i, (clase, sub) in enumerate(df.groupby("clase")):
        ax.scatter(
            sub["dim1"], sub["dim2"], s=10, alpha=0.65,
            label=str(clase), color=palette[i % len(palette)],
        )
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(fontsize=7, framealpha=0.85)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.18)
    return fig

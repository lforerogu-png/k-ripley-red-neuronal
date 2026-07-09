"""Utilidades de visualización (matplotlib).

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
from matplotlib.colors import ListedColormap, BoundaryNorm

from . import COLOR_AMBOS, COLOR_BLANCA, COLOR_SOLO_A, COLOR_SOLO_B


def matriz_patron(p, n_grid: int = 30) -> np.ndarray:
    """Matriz binaria 0/1: 1 si la celda está ocupada por el patrón."""
    mat = np.zeros((n_grid, n_grid), dtype=int)
    if len(p):
        for c, r in zip(p["col"], p["row"]):
            if 1 <= c <= n_grid and 1 <= r <= n_grid:
                mat[int(r) - 1, int(c) - 1] = 1
    return mat


def _dibujar_grilla_celdas(ax, n_grid: int, color: str = "gray", lw: float = 0.25):
    """Líneas de separación entre celdas (borde de cada cuadrado de la grilla)."""
    for i in range(n_grid + 1):
        y = 0.5 + i
        ax.axhline(y, color=color, linewidth=lw, zorder=2)
        ax.axvline(y, color=color, linewidth=lw, zorder=2)


def fig_patron(p, color, titulo, n_grid=30):
    """Dibuja un patrón como cuadrados rellenos sobre grilla N×N.

    Mismo estilo que :func:`fig_grilla_colores`: ``imshow`` sobre matriz
    discreta, una celda = un cuadrado del color del patrón, fondo blanco y
    líneas grises finas entre celdas.
    """
    ocupacion = matriz_patron(p, n_grid)
    cmap = ListedColormap([COLOR_BLANCA, color])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
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
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    fig.tight_layout()
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
    # 1 -> solo A, 2 -> solo B, 3 -> ambos
    return estado


def fig_grilla_colores(p1, p2, n_grid=30, titulo="Superposición (4 colores)"):
    """Grilla con los 4 colores: blanco/azul/amarillo/verde."""
    estado = matriz_estados(p1, p2, n_grid)
    cmap = ListedColormap([COLOR_BLANCA, COLOR_SOLO_A, COLOR_SOLO_B, COLOR_AMBOS])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
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
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    from matplotlib.patches import Patch
    leyenda = [
        Patch(facecolor=COLOR_BLANCA, edgecolor="gray", label="Blanca (ninguno)"),
        Patch(facecolor=COLOR_SOLO_A, label="Solo A (azul)"),
        Patch(facecolor=COLOR_SOLO_B, label="Solo B (amarillo)"),
        Patch(facecolor=COLOR_AMBOS, label="Ambos (verde)"),
    ]
    ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def fig_curva_k(res, titulo, color="#1565C0", envolvente=None, r_ref=0.2):
    """Curva K observada vs teórica (y envolvente opcional)."""
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.plot(res.r, res.k, color=color, lw=2, label="K observada")
    ax.plot(res.r, res.k_teorica, "--", color="gray", lw=1.2, label="K teórica (CSR)")
    if envolvente is not None:
        ax.fill_between(envolvente.r, envolvente.lo, envolvente.hi,
                        color="gray", alpha=0.25, label="Envolvente CSR")
    ax.axvline(r_ref, color="red", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("Radio r"); ax.set_ylabel("K(r)")
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_curvas_aprendizaje(resultado, titulo="Curvas de aprendizaje"):
    """Curvas de pérdida train/val por época."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(resultado.loss_train, label="Pérdida train", color="#1565C0")
    if resultado.loss_val:
        ax.plot(resultado.loss_val, label="Pérdida val", color="#C62828")
    ax.set_xlabel("Época"); ax.set_ylabel("Log-loss (entropía cruzada)")
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_matriz_confusion(cm: pd.DataFrame, titulo="Matriz de confusión"):
    """Heatmap de una matriz de confusión (DataFrame)."""
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    data = cm.to_numpy()
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels([c.replace("pred_", "") for c in cm.columns],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels([i.replace("real_", "") for i in cm.index], fontsize=8)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    umbral = data.max() / 2 if data.max() else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, int(data[i, j]), ha="center", va="center",
                    color="white" if data[i, j] > umbral else "black", fontsize=9)
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def fig_lisa(cluster: np.ndarray, n_grid: int, titulo="Clústeres LISA"):
    """Mapa de clústeres LISA sobre la grilla."""
    codigo = {"ns": 0, "HH": 1, "LL": 2, "HL": 3, "LH": 4}
    colores = ["#eeeeee", "#c62828", "#1565c0", "#f9a825", "#6a1b9a"]
    mat = np.array([codigo.get(c, 0) for c in cluster]).reshape(n_grid, n_grid)
    cmap = ListedColormap(colores)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(mat, cmap=cmap, norm=norm, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    from matplotlib.patches import Patch
    leyenda = [Patch(facecolor=colores[v], label=k) for k, v in codigo.items()]
    ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.05),
              ncol=5, fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def fig_dispersion_2d(df, titulo="Proyección 2D"):
    """Scatter 2D coloreado por clase (para PCA/t-SNE)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for clase, sub in df.groupby("clase"):
        ax.scatter(sub["dim1"], sub["dim2"], s=8, alpha=0.6, label=str(clase))
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig

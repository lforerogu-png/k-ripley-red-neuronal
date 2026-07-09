"""App Streamlit: K de Ripley cruzada y Perceptrón Multicapa.

Interfaz académica para análisis de coocurrencia espacial en epidemiología
vegetal. Las métricas se calculan solo sobre celdas coloreadas (espacio
condicional).

Ejecutar con:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coocurrencia import (  # noqa: E402
    CLASES_2, CLASES_3, dataset, metrics, models, ripley, viz,
)
from coocurrencia.patterns import (  # noqa: E402
    contar_coincidencias, generar_patron, generar_patron_condicionado,
    rango_coincidencias,
)

st.set_page_config(
    page_title="Ripley K — Spatial Cooccurrence",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Estilos globales
# ---------------------------------------------------------------------------
def _inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg-main: #0f1117;
            --bg-panel: #1a1d27;
            --bg-sidebar: #13151f;
            --accent: #4a9eff;
            --accent-light: #64b5f6;
            --positive: #66bb6a;
            --warning: #ffa726;
            --negative: #ef5350;
            --text-primary: #dde1f0;
            --text-secondary: #7b82a0;
            --border: #252836;
            --formula-bg: #1e2d4a;
        }

        .stApp, [data-testid="stAppViewContainer"] {
            background-color: var(--bg-main);
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
        }

        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar);
            min-width: 300px !important;
            max-width: 300px !important;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] .block-container {
            padding: 1rem 1.1rem;
        }

        .main .block-container {
            padding: 24px 24px 2rem 24px;
            max-width: 100%;
        }

        h1, h2, h3, h4, p, label, span, li {
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-size: 0.68rem !important;
            font-weight: 600 !important;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--text-secondary) !important;
            margin: 1.25rem 0 0.6rem 0 !important;
            padding-bottom: 0.45rem;
            border-bottom: 1px solid var(--border);
        }

        [data-testid="stSidebar"] h1:first-of-type {
            margin-top: 0 !important;
        }

        .app-header {
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border);
            padding: 1.25rem 24px;
            margin: -24px -24px 24px -24px;
        }
        .app-header h1 {
            font-size: 1.45rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
        }
        .app-header .subtitle {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin: 0;
            line-height: 1.5;
        }
        .app-header .status-line {
            font-size: 0.78rem;
            color: var(--text-secondary);
            margin-top: 0.65rem;
            font-family: 'JetBrains Mono', monospace;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: transparent;
            border-bottom: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: var(--text-secondary) !important;
            font-size: 0.85rem;
            font-weight: 500;
            padding: 0.65rem 1.1rem;
            border: none !important;
            border-radius: 0 !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--text-primary) !important;
            border-bottom: 2px solid var(--accent) !important;
            background: transparent !important;
        }

        .metric-card {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent-color, var(--accent));
            border-radius: 6px;
            padding: 1rem 1.1rem;
            transition: border-color 0.2s ease;
            min-height: 88px;
        }
        .metric-card:hover {
            border-color: var(--accent-light);
        }
        .metric-card .label {
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin-bottom: 0.35rem;
        }
        .metric-card .value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 500;
            color: var(--text-primary);
            line-height: 1.1;
        }
        .metric-card .delta {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        .plain-text {
            font-size: 0.85rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }
        .plain-text strong { color: var(--text-primary); font-weight: 500; }

        .cell-counts {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-align: center;
            padding: 0.75rem 0;
            border-top: 1px solid var(--border);
            margin-top: 0.5rem;
        }

        .section-title {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin: 1.5rem 0 0.75rem 0;
            padding-bottom: 0.4rem;
            border-bottom: 1px solid var(--border);
        }

        .notice-box {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 4px;
            padding: 0.85rem 1rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin: 0.75rem 0;
            line-height: 1.55;
        }

        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            margin: 0.5rem 0 1rem 0;
        }
        .metrics-table th {
            text-align: left;
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: var(--text-secondary);
            padding: 0.55rem 0.75rem;
            border-bottom: 1px solid var(--border);
        }
        .metrics-table td {
            padding: 0.5rem 0.75rem;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
        }
        .metrics-table tr:nth-child(odd) td { background: #1a1d27; }
        .metrics-table tr:nth-child(even) td { background: #1f2235; }
        .val-high { color: var(--positive) !important; }
        .val-mid { color: var(--warning) !important; }
        .val-low { color: var(--negative) !important; }

        .formula-block {
            background: var(--formula-bg);
            border-left: 3px solid var(--accent);
            padding: 0.85rem 1rem;
            margin: 1rem 0;
            border-radius: 0 4px 4px 0;
            font-size: 0.9rem;
        }

        .theory-body {
            color: var(--text-secondary);
            font-size: 0.9rem;
            line-height: 1.75;
            max-width: 820px;
        }
        .theory-body h3 {
            color: var(--text-primary);
            font-size: 1rem;
            font-weight: 600;
            margin: 1.75rem 0 0.65rem 0;
        }
        .theory-body p { margin-bottom: 0.85rem; }

        .interp-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            margin: 1rem 0;
        }
        .interp-table th, .interp-table td {
            border: 1px solid var(--border);
            padding: 0.6rem 0.75rem;
            text-align: left;
        }
        .interp-table th {
            background: var(--bg-panel);
            color: var(--text-secondary);
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }
        .interp-table td { color: var(--text-secondary); }
        .interp-table td:first-child {
            color: var(--text-primary);
            font-weight: 500;
        }

        .app-footer {
            border-top: 1px solid var(--border);
            margin-top: 2.5rem;
            padding: 1.25rem 24px;
            text-align: center;
            font-size: 0.72rem;
            color: var(--text-secondary);
            margin-left: -24px;
            margin-right: -24px;
        }

        .stSlider [data-baseweb="slider"] div {
            color: var(--accent) !important;
        }
        div[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            background: var(--accent) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 4px !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            padding: 0.55rem 1rem !important;
            box-shadow: none !important;
        }
        div[data-testid="stSidebar"] .stButton > button:hover {
            background: var(--accent-light) !important;
        }

        [data-testid="stAlert"] {
            background: var(--bg-panel) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-secondary) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_section(title: str):
    st.sidebar.markdown(f"### {title}")


def _metric_card(label: str, value: str, delta: str = "", accent: str = "#4a9eff"):
    st.markdown(
        f"""
        <div class="metric-card" style="--accent-color: {accent};">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {"<div class='delta'>" + delta + "</div>" if delta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _color_metrica(val: float) -> str:
    if val > 0.9:
        return "val-high"
    if val >= 0.7:
        return "val-mid"
    return "val-low"


def _tabla_metricas_html(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    rows_html = []
    for idx, (_, row) in enumerate(df.iterrows()):
        cells = [f"<td>{idx}</td>"]
        for c in cols:
            v = row[c]
            if isinstance(v, (int, float, np.floating, np.integer)):
                cls = _color_metrica(float(v))
                cells.append(f'<td class="{cls}">{v:.3f}</td>')
            else:
                cells.append(f"<td>{v}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    header = "<th>clase</th>" + "".join(f"<th>{c}</th>" for c in cols)
    return (
        '<table class="metrics-table"><thead><tr>'
        + header
        + "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )


def _estado_app(usar_a: bool, usar_b: bool) -> str:
    if usar_a and usar_b:
        return "Modo: datos observados (patrones A y B)"
    if usar_a:
        return "Modo: datos observados (patrón A) + simulación (patrón B)"
    if usar_b:
        return "Modo: simulación (patrón A) + datos observados (patrón B)"
    return "Modo: simulación estocástica (CSR / condicionado)"


def _accent_clasificacion(cls: str) -> str:
    m = {
        "agrupado": "#4a9eff", "atraccion": "#66bb6a",
        "aleatorio": "#7b82a0", "sin_coocurrencia": "#7b82a0",
        "disperso": "#ffa726", "repulsion": "#ef5350",
    }
    return m.get(str(cls).lower(), "#4a9eff")


def _etiqueta_clase(cls: str) -> str:
    return {
        "agrupado": "Agrupado", "disperso": "Disperso", "aleatorio": "Aleatorio",
        "atraccion": "Atracción", "repulsion": "Repulsión",
        "sin_coocurrencia": "Independencia",
    }.get(str(cls).lower(), str(cls).capitalize())


# ---------------------------------------------------------------------------
# Datos: CSV y patrones cacheados
# ---------------------------------------------------------------------------
def _generar_csv_ejemplo() -> bytes:
    rng = np.random.default_rng(2024)
    df_ejemplo = pd.DataFrame({
        "x": np.round(rng.uniform(0, 30, size=60), 2),
        "y": np.round(rng.uniform(0, 30, size=60), 2),
    })
    return df_ejemplo.to_csv(index=False).encode("utf-8")


def _formatear_coord(valor: float) -> str:
    """Formatea un valor de coordenada para mostrar en mensajes al usuario."""
    if not np.isfinite(valor):
        return "nan"
    if abs(valor) >= 1000 or (abs(valor) > 0 and abs(valor) < 0.01):
        return f"{valor:.4g}"
    return f"{valor:.4f}".rstrip("0").rstrip(".")


def _rescale_minmax(vals: np.ndarray, vmin: float, vmax: float, n_grid: int) -> np.ndarray:
    """Re-escala linealmente ``vals`` al rango [0, ``n_grid``]."""
    if vmax == vmin:
        return np.full_like(vals, n_grid / 2.0, dtype=float)
    return (vals - vmin) / (vmax - vmin) * n_grid


def _procesar_csv_subido(archivo, n_grid: int) -> dict:
    resultado: dict = {"error": None, "warning": None, "puntos": None, "info": None}
    try:
        df = pd.read_csv(archivo)
        if df is None or df.empty:
            resultado["error"] = "El archivo está vacío."
            return resultado
        columnas = {str(c).strip().lower(): c for c in df.columns}
        if "x" not in columnas or "y" not in columnas:
            resultado["error"] = "El archivo debe tener columnas 'x' e 'y'"
            return resultado
        x = pd.to_numeric(df[columnas["x"]], errors="coerce").to_numpy()
        y = pd.to_numeric(df[columnas["y"]], errors="coerce").to_numpy()
        validos = np.isfinite(x) & np.isfinite(y)
        x, y = x[validos], y[validos]
        if len(x) == 0:
            resultado["error"] = "No se encontraron coordenadas numéricas válidas."
            return resultado

        min_x, max_x = float(x.min()), float(x.max())
        min_y, max_y = float(y.min()), float(y.max())
        x_celda = _rescale_minmax(x, min_x, max_x, n_grid)
        y_celda = _rescale_minmax(y, min_y, max_y, n_grid)

        col = np.clip(np.floor(x_celda).astype(int) + 1, 1, n_grid)
        row = np.clip(np.floor(y_celda).astype(int) + 1, 1, n_grid)
        puntos = (
            pd.DataFrame({"col": col, "row": row})
            .drop_duplicates()
            .reset_index(drop=True)
        )
        if len(puntos) == 0:
            resultado["error"] = "No quedaron puntos válidos tras deduplicar."
            return resultado
        if len(puntos) > 500:
            puntos = puntos.iloc[:500].reset_index(drop=True)
            resultado["warning"] = "Se truncaron a 500 puntos por rendimiento."
        elif len(puntos) < 10:
            resultado["warning"] = "Se recomiendan al menos 10 puntos para K de Ripley."
        puntos["x"] = (puntos["col"] - 0.5) / n_grid
        puntos["y"] = (puntos["row"] - 0.5) / n_grid
        resultado["puntos"] = puntos
        resultado["info"] = {
            "n": len(puntos),
            "mensaje": (
                f"Coordenadas originales: x=[{_formatear_coord(min_x)}, "
                f"{_formatear_coord(max_x)}], y=[{_formatear_coord(min_y)}, "
                f"{_formatear_coord(max_y)}]. Re-escaladas automáticamente a la "
                f"grilla {n_grid}×{n_grid}."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        resultado["error"] = f"No se pudo procesar el archivo: {exc}"
    return resultado


@st.cache_data(show_spinner=False)
def _patron_simulado(tipo, n, seed, n_grid):
    return generar_patron(tipo, n, seed, n_grid=n_grid)


@st.cache_data(show_spinner=False)
def _patron_condicionado(p1, tipo2, n2, n_coincidentes, seed2, n_grid):
    return generar_patron_condicionado(p1, tipo2, n2, n_coincidentes, seed2, n_grid=n_grid)


def _ejecutar_entrenamiento(
    n_sim, n_grid, r_ref, metodo_k, pct_train, capa, lr,
    p1_fijo, p2_fijo,
):
    datos = dataset.simular_dataset(
        n_sim=n_sim, n_grid=n_grid, n_pts_min=60, n_pts_max=200,
        r_ref=r_ref, metodo=metodo_k, seed=42,
        p1_fijo=p1_fijo, p2_fijo=p2_fijo,
    )
    if datos.empty:
        raise ValueError("No se pudo generar el dataset de entrenamiento.")
    coloreadas = dataset.filtrar_coloreadas(datos)
    if coloreadas["clase_celda"].nunique() < 2:
        st.session_state["train_warning"] = (
            "El dataset contiene una sola clase coloreada; las métricas serán limitadas."
        )
    fracs = (pct_train / 100, (1 - pct_train / 100) / 2, (1 - pct_train / 100) / 2)
    part = models.particion_por_sim(datos, fracs, seed=42)
    hidden = eval(capa)
    res = models.entrenar_mlp(
        part.train, part.val, hidden_layer_sizes=hidden,
        learning_rate_init=lr, max_epocas=250,
    )
    pred = res.predecir(part.test)
    proba = res.predecir_proba(part.test)
    n_blancas = int(datos[datos["sim_id"].isin(
        part.test["sim_id"].unique())]["celda_blanca"].sum())
    rep = metrics.evaluar(
        part.test["clase_celda"].to_numpy(), pred, proba,
        clases=CLASES_3, n_blancas=n_blancas,
    )
    st.session_state["rep"] = rep
    st.session_state["res"] = res
    st.session_state["datos_info"] = {
        "n_train": len(part.train), "n_test": len(part.test),
        "esquema": part.esquema,
    }


# ---------------------------------------------------------------------------
# Inicialización visual
# ---------------------------------------------------------------------------
_inject_css()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Controles")

_sidebar_section("Grilla")
n_grid = st.sidebar.slider(
    "Tamaño de grilla (N × N)", 10, 50, 30, 5, key="n_grid",
)

_sidebar_section("Datos observados — Patrón A")
archivo_csv_a = st.sidebar.file_uploader(
    "Archivo CSV (coordenadas A)", type=["csv"], key="csv_uploader_a",
)
st.sidebar.caption(
    "Columnas requeridas: x, y. Cualquier unidad o rango; se re-escalan automáticamente a la grilla."
)
st.sidebar.download_button(
    "Descargar plantilla CSV",
    data=_generar_csv_ejemplo(),
    file_name="plantilla_puntos.csv",
    mime="text/csv",
    key="descargar_ejemplo_a",
)

puntos_subidos_a: Optional[pd.DataFrame] = None
usar_datos_propios_a = False
if archivo_csv_a is not None:
    resultado_csv_a = _procesar_csv_subido(archivo_csv_a, n_grid)
    if resultado_csv_a["error"]:
        st.sidebar.error(resultado_csv_a["error"])
    else:
        puntos_subidos_a = resultado_csv_a["puntos"]
        if resultado_csv_a["warning"]:
            st.sidebar.warning(resultado_csv_a["warning"])
        info_a = resultado_csv_a["info"]
        st.sidebar.caption(f"{info_a['n']} puntos cargados.")
        st.sidebar.info(info_a["mensaje"])
    usar_datos_propios_a = st.sidebar.checkbox(
        "Usar datos observados (A)",
        value=puntos_subidos_a is not None,
        disabled=puntos_subidos_a is None,
        key="usar_propios_a",
    )

_sidebar_section("Datos observados — Patrón B")
archivo_csv_b = st.sidebar.file_uploader(
    "Archivo CSV (coordenadas B)", type=["csv"], key="csv_uploader_b",
)
st.sidebar.caption(
    "Columnas requeridas: x, y. Cualquier unidad o rango; se re-escalan automáticamente a la grilla."
)
st.sidebar.download_button(
    "Descargar plantilla CSV",
    data=_generar_csv_ejemplo(),
    file_name="plantilla_puntos.csv",
    mime="text/csv",
    key="descargar_ejemplo_b",
)

puntos_subidos_b: Optional[pd.DataFrame] = None
usar_datos_propios_b = False
if archivo_csv_b is not None:
    resultado_csv_b = _procesar_csv_subido(archivo_csv_b, n_grid)
    if resultado_csv_b["error"]:
        st.sidebar.error(resultado_csv_b["error"])
    else:
        puntos_subidos_b = resultado_csv_b["puntos"]
        if resultado_csv_b["warning"]:
            st.sidebar.warning(resultado_csv_b["warning"])
        info_b = resultado_csv_b["info"]
        st.sidebar.caption(f"{info_b['n']} puntos cargados.")
        st.sidebar.info(info_b["mensaje"])
    usar_datos_propios_b = st.sidebar.checkbox(
        "Usar datos observados (B)",
        value=puntos_subidos_b is not None,
        disabled=puntos_subidos_b is None,
        key="usar_propios_b",
    )

_sidebar_section("Patrón A")
tipo1 = st.sidebar.selectbox(
    "Distribución espacial (A)", ["agrupado", "disperso", "aleatorio"], 0,
    disabled=usar_datos_propios_a, key="tipo_a",
)
n1 = st.sidebar.slider(
    "Número de puntos (A)", 1, 500, 120, 1,
    disabled=usar_datos_propios_a, key="n_puntos_a",
)
seed1 = st.sidebar.number_input(
    "Semilla aleatoria (A)", 1, 999999, 101,
    disabled=usar_datos_propios_a, key="seed_a",
)

_sidebar_section("Patrón B")
tipo2 = st.sidebar.selectbox(
    "Distribución espacial (B)", ["agrupado", "disperso", "aleatorio"], 1,
    disabled=usar_datos_propios_b, key="tipo_b",
)
n2 = st.sidebar.slider(
    "Número de puntos (B)", 1, 500, 120, 1,
    disabled=usar_datos_propios_b, key="n_puntos_b",
)
n_efectivo_1 = (
    len(puntos_subidos_a) if (usar_datos_propios_a and puntos_subidos_a is not None) else n1
)
min_coinc, max_coinc = rango_coincidencias(n_efectivo_1, n2, n_grid)
if min_coinc >= max_coinc:
    # Rango degenerado: A y B (más la grilla) dejan un único valor válido.
    n_coincidentes = max_coinc
    st.sidebar.caption(f"Celdas coincidentes (B|A): fijo en {max_coinc} (sin margen posible).")
else:
    valor_default = int(np.clip(30, min_coinc, max_coinc))
    n_coincidentes = st.sidebar.slider(
        "Celdas coincidentes (B|A)", min_coinc, max_coinc, valor_default, 1,
        disabled=usar_datos_propios_b,
        help="No aplica con datos observados en B.",
        key="n_coincidentes",
    )
if min_coinc > 0 and not usar_datos_propios_b:
    st.sidebar.caption(
        f"Mínimo obligatorio: {min_coinc} · A ({n_efectivo_1}) + B ({n2}) "
        f"excede la capacidad de la grilla ({n_grid}² = {n_grid * n_grid})."
    )
seed2 = st.sidebar.number_input(
    "Semilla aleatoria (B)", 1, 999999, 202,
    disabled=usar_datos_propios_b, key="seed_b",
)

_sidebar_section("K de Ripley")
r_ref = st.sidebar.slider(
    "Radio de referencia r", 0.05, 0.25, 0.20, 0.01, key="r_ref",
)
metodo_k = st.sidebar.radio(
    "Criterio de clasificación",
    ["umbral", "envolvente"],
    index=0,
    help="Umbral fijo (πr²) o envolventes Monte Carlo bajo CSR.",
    key="metodo_k",
)

_sidebar_section("Red neuronal")
n_sim = st.sidebar.slider("Simulaciones", 30, 400, 150, 10, key="n_sim")
pct_train = st.sidebar.slider(
    "Proporción entrenamiento (%)", 50, 90, 70, 5, key="pct_train",
)
capa = st.sidebar.selectbox(
    "Arquitectura oculta",
    ["(8,)", "(16,)", "(16, 8)", "(32, 16)", "(32, 16, 8)"], 2,
    key="capa",
)
lr = st.sidebar.select_slider(
    "Tasa de aprendizaje", [0.001, 0.01, 0.1], 0.01, key="lr",
)

st.sidebar.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
entrenar_click = st.sidebar.button("Entrenar modelo", type="primary", key="btn_entrenar")

# ---------------------------------------------------------------------------
# Patrones y resumen K
# ---------------------------------------------------------------------------
if usar_datos_propios_a and puntos_subidos_a is not None:
    p1 = puntos_subidos_a
else:
    p1 = _patron_simulado(tipo1, n1, seed1, n_grid)

if usar_datos_propios_b and puntos_subidos_b is not None:
    p2 = puntos_subidos_b
else:
    p2 = _patron_condicionado(p1, tipo2, n2, n_coincidentes, seed2, n_grid)

resumen = dataset.analizar_par(p1, p2, r_ref=r_ref, metodo=metodo_k)
p1_fijo_mlp = p1 if (usar_datos_propios_a and puntos_subidos_a is not None) else None
p2_fijo_mlp = p2 if (usar_datos_propios_b and puntos_subidos_b is not None) else None

if entrenar_click:
    with st.spinner("Entrenando modelo..."):
        try:
            _ejecutar_entrenamiento(
                n_sim, n_grid, r_ref, metodo_k, pct_train, capa, lr,
                p1_fijo_mlp, p2_fijo_mlp,
            )
        except Exception as exc:  # noqa: BLE001
            st.session_state["train_error"] = str(exc)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <h1>Ripley K — Análisis de coocurrencia espacial</h1>
        <p class="subtitle">
            Estimación de funciones K univariadas y cruzadas con perceptrón multicapa.
            Evaluación condicional sobre celdas con al menos un evento.
        </p>
        <p class="status-line">{_estado_app(usar_datos_propios_a, usar_datos_propios_b)}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pat, tab_k, tab_grilla, tab_mlp, tab_teoria = st.tabs([
    "Patrones",
    "Funciones K",
    "Grilla",
    "Modelo",
    "Referencia",
])

# ---------------------------------------------------------------------------
# Tab: Patrones
# ---------------------------------------------------------------------------
with tab_pat:
    if usar_datos_propios_a and puntos_subidos_a is not None:
        titulo_a = f"Patrón A — datos observados ({len(p1)} pts)"
    else:
        titulo_a = f"Patrón A — {tipo1}"
    if usar_datos_propios_b and puntos_subidos_b is not None:
        titulo_b = f"Patrón B — datos observados ({len(p2)} pts)"
    else:
        titulo_b = f"Patrón B — {tipo2}"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(viz.fig_patron(p1, "#4a9eff", titulo_a, n_grid), use_container_width=True)
    with c2:
        st.pyplot(viz.fig_patron(p2, "#ffa726", titulo_b, n_grid), use_container_width=True)
    with c3:
        st.pyplot(
            viz.fig_grilla_colores(
                p1, p2, n_grid, "Superposición",
                resultado=resumen.clase_biv,
            ),
            use_container_width=True,
        )

    st.markdown('<p class="section-title">Clasificación univariada y bivariada</p>',
                unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        _metric_card(
            "Patrón A", _etiqueta_clase(resumen.cls1),
            f"K = {resumen.k1_val:.4f}", _accent_clasificacion(resumen.cls1),
        )
    with m2:
        _metric_card(
            "Patrón B", _etiqueta_clase(resumen.cls2),
            f"K = {resumen.k2_val:.4f}", _accent_clasificacion(resumen.cls2),
        )
    with m3:
        _metric_card(
            "Coocurrencia K₁₂", _etiqueta_clase(resumen.clase_biv),
            f"K₁₂ = {resumen.k12_val:.4f}", _accent_clasificacion(resumen.clase_biv),
        )

    n_coinc_real = contar_coincidencias(p1, p2)
    st.markdown(
        f"""
        <p class="plain-text">
        Celdas coincidentes: <strong>{n_coinc_real}</strong>
        &nbsp;·&nbsp; Referencia CSR en r = {r_ref}: πr² = {np.pi * r_ref**2:.4f}
        &nbsp;·&nbsp; Criterio: {metodo_k}
        </p>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tab: Funciones K
# ---------------------------------------------------------------------------
with tab_k:
    envs = {"k1": None, "k2": None, "k12": None}
    if metodo_k == "envolvente":
        a = p1[["x", "y"]].to_numpy()
        b = p2[["x", "y"]].to_numpy()
        envs["k1"] = ripley.envolvente_csr(len(a), n_sim=39, seed=1)
        envs["k2"] = ripley.envolvente_csr(len(b), n_sim=39, seed=2)
        if len(a) and len(b):
            envs["k12"] = ripley.envolvente_independencia(a, b, n_sim=39, seed=3)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(viz.fig_curva_k(
            resumen.k1_curva, "K univariada — A", clasificacion=resumen.cls1,
            envolvente=envs["k1"], r_ref=r_ref,
        ), use_container_width=True)
        st.pyplot(viz.fig_curva_l(
            ripley.k_a_l(resumen.k1_curva), "L univariada — A",
            clasificacion=resumen.cls1,
            envolvente=ripley.envolvente_k_a_l(envs["k1"]) if envs["k1"] else None,
            r_ref=r_ref,
        ), use_container_width=True)
    with c2:
        st.pyplot(viz.fig_curva_k(
            resumen.k2_curva, "K univariada — B", color="#ffa726",
            clasificacion=resumen.cls2, envolvente=envs["k2"], r_ref=r_ref,
        ), use_container_width=True)
        st.pyplot(viz.fig_curva_l(
            ripley.k_a_l(resumen.k2_curva), "L univariada — B", color="#ffa726",
            clasificacion=resumen.cls2,
            envolvente=ripley.envolvente_k_a_l(envs["k2"]) if envs["k2"] else None,
            r_ref=r_ref,
        ), use_container_width=True)
    with c3:
        st.pyplot(viz.fig_curva_k(
            resumen.k12_curva, "K cruzada — K₁₂", color="#66bb6a",
            clasificacion=resumen.clase_biv, envolvente=envs["k12"], r_ref=r_ref,
        ), use_container_width=True)
        st.pyplot(viz.fig_curva_l(
            ripley.k_a_l(resumen.k12_curva), "L cruzada — L₁₂", color="#66bb6a",
            clasificacion=resumen.clase_biv,
            envolvente=ripley.envolvente_k_a_l(envs["k12"]) if envs["k12"] else None,
            r_ref=r_ref,
        ), use_container_width=True)

    st.markdown(
        """
        <p class="plain-text">
        <strong>K &gt; πr²</strong> / <strong>L &gt; 0</strong> — agrupamiento o atracción &nbsp;·&nbsp;
        <strong>K ≈ πr²</strong> / <strong>L ≈ 0</strong> — aleatoriedad (CSR) o independencia &nbsp;·&nbsp;
        <strong>K &lt; πr²</strong> / <strong>L &lt; 0</strong> — dispersión o repulsión
        </p>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tab: Grilla
# ---------------------------------------------------------------------------
with tab_grilla:
    estado = viz.matriz_estados(p1, p2, n_grid)
    n_blanca = int(np.sum(estado == 0))
    n_azul = int(np.sum(estado == 1))
    n_amar = int(np.sum(estado == 2))
    n_verde = int(np.sum(estado == 3))

    st.markdown(
        f'<p class="plain-text">Resultado: <strong>{_etiqueta_clase(resumen.clase_biv)}</strong></p>',
        unsafe_allow_html=True,
    )
    st.pyplot(
        viz.fig_grilla_colores(
            p1, p2, n_grid, "Grilla de coocurrencia",
            resultado=resumen.clase_biv,
        ),
        use_container_width=False,
    )
    st.markdown(
        f"""
        <p class="cell-counts">
        Solo A: {n_azul} &nbsp;|&nbsp; Solo B: {n_amar} &nbsp;|&nbsp;
        Ambos: {n_verde} &nbsp;|&nbsp; Sin eventos: {n_blanca}
        </p>
        <p class="plain-text" style="margin-top:0.5rem;">
        Las celdas sin evento se excluyen del espacio de evaluación del modelo
        y se reportan únicamente con fines descriptivos.
        </p>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tab: Modelo (MLP)
# ---------------------------------------------------------------------------
with tab_mlp:
    st.markdown(
        """
        <p class="plain-text">
        Entrenamiento del perceptrón multicapa sobre simulaciones Monte Carlo.
        La evaluación se restringe a celdas coloreadas (espacio condicional).
        Use el botón <strong>Entrenar modelo</strong> en el panel lateral.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if p1_fijo_mlp is not None and p2_fijo_mlp is not None:
        st.markdown(
            f'<div class="notice-box">Par fijo: {len(p1_fijo_mlp)} puntos (A), '
            f'{len(p2_fijo_mlp)} puntos (B). Sin variabilidad entre simulaciones.</div>',
            unsafe_allow_html=True,
        )
    elif p1_fijo_mlp is not None:
        st.markdown(
            f'<div class="notice-box">Patrón A fijo ({len(p1_fijo_mlp)} puntos). '
            f'El patrón B varía entre simulaciones.</div>',
            unsafe_allow_html=True,
        )
    elif p2_fijo_mlp is not None:
        st.markdown(
            f'<div class="notice-box">Patrón B fijo ({len(p2_fijo_mlp)} puntos). '
            f'El patrón A varía entre simulaciones.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.get("train_error"):
        st.error(st.session_state.pop("train_error"))
    if st.session_state.get("train_warning"):
        st.warning(st.session_state.pop("train_warning"))

    if "rep" in st.session_state:
        rep = st.session_state["rep"]
        info = st.session_state["datos_info"]

        st.markdown('<p class="section-title">Resumen de evaluación</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            acc = rep.accuracy
            _metric_card(
                "Accuracy condicional", f"{acc*100:.1f}%",
                accent="#66bb6a" if acc > 0.9 else "#ffa726" if acc >= 0.7 else "#ef5350",
            )
        with c2:
            f1 = rep.macro["f1"]
            _metric_card(
                "Macro F1", f"{f1:.3f}",
                accent="#66bb6a" if f1 > 0.9 else "#ffa726" if f1 >= 0.7 else "#ef5350",
            )
        with c3:
            _metric_card("Celdas excluidas", str(rep.n_blancas), "sin evento")

        st.markdown('<p class="section-title">Métricas por clase</p>', unsafe_allow_html=True)
        st.markdown(_tabla_metricas_html(rep.por_clase.round(3)), unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            st.pyplot(viz.fig_matriz_confusion(rep.matriz_3x3, "Matriz 3 × 3"))
        with cc2:
            st.pyplot(viz.fig_matriz_confusion(rep.matriz_2x2, "Matriz 2 × 2"))

        if rep.auc_ovr:
            auc_txt = " &nbsp;·&nbsp; ".join(
                f"{k}: {v:.3f}" for k, v in rep.auc_ovr.items()
            )
            st.markdown(
                f'<p class="plain-text">AUC-ROC (OvR): {auc_txt}</p>',
                unsafe_allow_html=True,
            )

        st.markdown('<p class="section-title">Curvas de aprendizaje</p>', unsafe_allow_html=True)
        st.pyplot(viz.fig_curvas_aprendizaje(st.session_state["res"]))
        st.markdown(
            f'<p class="plain-text">Partición {info["esquema"]} — '
            f'entrenamiento: {info["n_train"]} celdas · '
            f'prueba: {info["n_test"]} celdas</p>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Tab: Referencia (teoría)
# ---------------------------------------------------------------------------
with tab_teoria:
    st.markdown(
        """
        <div class="theory-body">
        <h3>Función K de Ripley univariada</h3>
        <p>
        La función K de Ripley (Ripley, 1976) estima la intensidad de puntos
        en discos de radio <em>r</em> centrados en cada evento del patrón.
        Para un proceso de Poisson homogéneo (completamente aleatorio, CSR),
        la referencia teórica es K(r) = πr².
        </p>
        <div class="formula-block">
        K(r) = λ⁻¹ E[N(r)] &nbsp;&nbsp; donde N(r) es el número de puntos
        a distancia ≤ r de un punto típico.
        </div>
        <p>
        Desviaciones respecto a πr² permiten clasificar el patrón como agrupado
        (K &gt; πr²), aleatorio (K ≈ πr²) o disperso (K &lt; πr²).
        </p>

        <h3>Función K cruzada (bivariada)</h3>
        <p>
        La K cruzada K₁₂(r) mide la asociación espacial entre dos tipos de
        puntos A y B (Diggle, 2003). Bajo independencia espacial, K₁₂(r) = πr².
        </p>
        <div class="formula-block">
        K₁₂(r) = (1/λ₂) E[N_B(r | punto A)] &nbsp;&nbsp; con corrección
        isotrópica de borde.
        </div>
        <p>
        Valores superiores a πr² indican atracción o coocurrencia positiva;
        inferiores, repulsión; cercanos, independencia espacial.
        </p>

        <h3>Evaluación condicional del modelo</h3>
        <p>
        Siguiendo Dixon (2002), las celdas sin ningún evento (blancas) se
        excluyen del cómputo de accuracy y matrices de confusión. Incluirlas
        inflaba la precisión al dominar la clase vacía. Se reportan aparte como
        información descriptiva para el productor.
        </p>

        <h3>Interpretación en contexto agronómico</h3>
        <table class="interp-table">
        <thead>
        <tr><th>Resultado K</th><th>Interpretación espacial</th><th>Implicación agronómica</th></tr>
        </thead>
        <tbody>
        <tr>
            <td>Agrupado / Atracción</td>
            <td>Los eventos tienden a co-localizarse más de lo esperado al azar.</td>
            <td>Posible foco de infección, gradiente ambiental o dispersión limitada del patógeno.</td>
        </tr>
        <tr>
            <td>Aleatorio / Independencia</td>
            <td>Distribución compatible con CSR o independencia entre capas.</td>
            <td>No hay evidencia de dependencia espacial detectable a la escala analizada.</td>
        </tr>
        <tr>
            <td>Disperso / Repulsión</td>
            <td>Los eventos se evitan mutuamente o aparecen de forma aislada.</td>
            <td>Puede indicar competencia, manejo del cultivo o exclusión por microambiente.</td>
        </tr>
        </tbody>
        </table>

        <h3>Referencias</h3>
        <p>
        Ripley, B. D. (1976). The second-order analysis of stationary point processes.
        <em>Journal of Applied Probability</em>, 13(2), 255–266.<br>
        Diggle, P. J. (2003). <em>Statistical Analysis of Spatial Point Patterns</em>
        (2nd ed.). Arnold.<br>
        Dixon, P. M. (2002). Ripley's K function. In <em>Encyclopedia of Environmetrics</em>
        (Vol. 3, pp. 1796–1803). Wiley.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Análisis de coocurrencia espacial en epidemiología vegetal —
        Universidad Nacional de Colombia
    </div>
    """,
    unsafe_allow_html=True,
)

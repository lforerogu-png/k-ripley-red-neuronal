"""App Streamlit: K de Ripley cruzada y Perceptrón Multicapa (versión Python).

Equivalente en espíritu a la app Shiny original, con la corrección
metodológica central: las métricas se calculan **solo sobre celdas
coloreadas** (espacio condicional). Las celdas blancas se reportan aparte
como información descriptiva.

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

# Permite ejecutar la app sin instalar el paquete (añade la raíz al path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coocurrencia import (  # noqa: E402
    CLASES_2, CLASES_3, dataset, metrics, models, ripley, viz,
)
from coocurrencia.patterns import (  # noqa: E402
    contar_coincidencias, generar_patron, generar_patron_condicionado,
)

st.set_page_config(page_title="K de Ripley + MLP", layout="wide")


# ---------------------------------------------------------------------------
# Funciones auxiliares: carga de datos propios (CSV) y patrones cacheados
# ---------------------------------------------------------------------------
def _generar_csv_ejemplo() -> bytes:
    """CSV de ejemplo: 60 puntos aleatorios en escala de celda (0-30)."""
    rng = np.random.default_rng(2024)
    df_ejemplo = pd.DataFrame({
        "x": np.round(rng.uniform(0, 30, size=60), 2),
        "y": np.round(rng.uniform(0, 30, size=60), 2),
    })
    return df_ejemplo.to_csv(index=False).encode("utf-8")


def _procesar_csv_subido(archivo, n_grid: int) -> dict:
    """Lee, valida y convierte un CSV de coordenadas (x, y) a celdas de la grilla.

    Devuelve un dict con claves ``error``, ``warning``, ``puntos`` (DataFrame
    con columnas ``col``, ``row``, ``x``, ``y``, listo para usarse como
    patrón A) e ``info`` (metadatos para el mensaje de éxito).
    """
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
            resultado["error"] = "No se encontraron coordenadas numéricas válidas en 'x'/'y'."
            return resultado

        normalizado = bool(np.all(np.abs(x) <= 1) and np.all(np.abs(y) <= 1))
        if normalizado:
            x_celda, y_celda = x * n_grid, y * n_grid
            escala_txt = "normalizada (0-1)"
        else:
            x_celda, y_celda = x, y
            escala_txt = "celdas (0-30)"

        col = np.clip(np.floor(x_celda).astype(int) + 1, 1, n_grid)
        row = np.clip(np.floor(y_celda).astype(int) + 1, 1, n_grid)
        puntos = (
            pd.DataFrame({"col": col, "row": row})
            .drop_duplicates()
            .reset_index(drop=True)
        )
        if len(puntos) == 0:
            resultado["error"] = "No quedaron puntos válidos después de eliminar duplicados."
            return resultado

        if len(puntos) > 500:
            puntos = puntos.iloc[:500].reset_index(drop=True)
            resultado["warning"] = (
                "Se usarán solo los primeros 500 puntos para mantener el rendimiento"
            )
        elif len(puntos) < 10:
            resultado["warning"] = (
                "Se recomienda tener al menos 10 puntos para un análisis confiable "
                "de K de Ripley"
            )

        puntos["x"] = (puntos["col"] - 0.5) / n_grid
        puntos["y"] = (puntos["row"] - 0.5) / n_grid

        resultado["puntos"] = puntos
        resultado["info"] = {
            "n": len(puntos),
            "escala": escala_txt,
            "rango": (
                f"x:[{x_celda.min():.1f}, {x_celda.max():.1f}]  "
                f"y:[{y_celda.min():.1f}, {y_celda.max():.1f}]"
            ),
        }
    except Exception as exc:  # noqa: BLE001 - captura cualquier archivo corrupto
        resultado["error"] = f"No se pudo procesar el archivo: {exc}"
    return resultado


@st.cache_data(show_spinner=False)
def _patron_simulado(tipo, n, seed, n_grid):
    return generar_patron(tipo, n, seed, n_grid=n_grid)


@st.cache_data(show_spinner=False)
def _patron_condicionado(p1, tipo2, n2, n_coincidentes, seed2, n_grid):
    return generar_patron_condicionado(p1, tipo2, n2, n_coincidentes, seed2, n_grid=n_grid)


# ---------------------------------------------------------------------------
# Barra lateral: controles
# ---------------------------------------------------------------------------
st.sidebar.title("Controles")

st.sidebar.header("Grilla")
n_grid = st.sidebar.slider("Tamaño de grilla (N x N)", 10, 50, 30, 5)

st.sidebar.header("📂 Cargar datos propios — Patrón A")
archivo_csv_a = st.sidebar.file_uploader(
    "Subir CSV con coordenadas (A)", type=["csv"], key="csv_uploader_a"
)
st.sidebar.caption(
    "El CSV debe tener columnas llamadas exactamente 'x' e 'y' con las "
    "coordenadas de los puntos. Valores entre 0 y 30 (coordenadas de celda) "
    "o entre 0 y 1 (normalizadas) — se detectan automáticamente."
)
st.sidebar.download_button(
    "Descargar CSV de ejemplo",
    data=_generar_csv_ejemplo(),
    file_name="ejemplo_puntos.csv",
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
        st.sidebar.success(
            f"Se cargaron {info_a['n']} puntos válidos. Escala detectada: "
            f"{info_a['escala']}. Rango: {info_a['rango']}."
        )
    usar_datos_propios_a = st.sidebar.checkbox(
        "Usar datos cargados en lugar del simulador (A)",
        value=puntos_subidos_a is not None,
        disabled=puntos_subidos_a is None,
        key="usar_propios_a",
    )

st.sidebar.header("📂 Cargar datos propios — Patrón B")
archivo_csv_b = st.sidebar.file_uploader(
    "Subir CSV con coordenadas (B)", type=["csv"], key="csv_uploader_b"
)
st.sidebar.caption(
    "El CSV debe tener columnas llamadas exactamente 'x' e 'y' con las "
    "coordenadas de los puntos. Valores entre 0 y 30 (coordenadas de celda) "
    "o entre 0 y 1 (normalizadas) — se detectan automáticamente."
)
st.sidebar.download_button(
    "Descargar CSV de ejemplo",
    data=_generar_csv_ejemplo(),
    file_name="ejemplo_puntos.csv",
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
        st.sidebar.success(
            f"Se cargaron {info_b['n']} puntos válidos. Escala detectada: "
            f"{info_b['escala']}. Rango: {info_b['rango']}."
        )
    usar_datos_propios_b = st.sidebar.checkbox(
        "Usar datos cargados en lugar del simulador (B)",
        value=puntos_subidos_b is not None,
        disabled=puntos_subidos_b is None,
        key="usar_propios_b",
    )

st.sidebar.header("Patrón A (azul)")
tipo1 = st.sidebar.selectbox(
    "Tipo A", ["agrupado", "disperso", "aleatorio"], 0, disabled=usar_datos_propios_a,
)
n1 = st.sidebar.slider(
    "Puntos A", 1, 500, 120, 1, disabled=usar_datos_propios_a,
)
seed1 = st.sidebar.number_input(
    "Semilla A", 1, 999999, 101, disabled=usar_datos_propios_a,
)

st.sidebar.header("Patrón B (amarillo)")
tipo2 = st.sidebar.selectbox(
    "Tipo B", ["agrupado", "disperso", "aleatorio"], 1, disabled=usar_datos_propios_b,
)
n2 = st.sidebar.slider(
    "Puntos B", 1, 500, 120, 1, disabled=usar_datos_propios_b,
)
n_efectivo_1 = (
    len(puntos_subidos_a) if (usar_datos_propios_a and puntos_subidos_a is not None) else n1
)
max_coinc = int(min(n_efectivo_1, n2))
n_coincidentes = st.sidebar.slider(
    "Celdas coincidentes B con A", 0, max_coinc, min(30, max_coinc), 1,
    disabled=usar_datos_propios_b,
    help="No aplica cuando B usa datos reales: las coincidencias se calculan "
         "directamente entre los puntos cargados.",
)
seed2 = st.sidebar.number_input(
    "Semilla B", 1, 999999, 202, disabled=usar_datos_propios_b,
)

st.sidebar.header("Clasificación K de Ripley")
r_ref = st.sidebar.slider("Radio de referencia r", 0.05, 0.25, 0.20, 0.01)
metodo_k = st.sidebar.radio(
    "Método de etiquetado", ["umbral", "envolvente"], index=0,
    help="Umbral fijo (pi*r^2) o envolventes de simulación Monte Carlo (CSR).",
)

st.sidebar.header("Red neuronal (MLP)")
n_sim = st.sidebar.slider("Simulaciones de entrenamiento", 30, 400, 150, 10)
pct_train = st.sidebar.slider("% entrenamiento", 50, 90, 70, 5)
capa = st.sidebar.selectbox(
    "Arquitectura oculta",
    ["(8,)", "(16,)", "(16, 8)", "(32, 16)", "(32, 16, 8)"], 2,
)
lr = st.sidebar.select_slider("Tasa de aprendizaje", [0.001, 0.01, 0.1], 0.01)


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("K de Ripley cruzada y Perceptrón Multicapa")
st.caption(
    "Coocurrencia en patrones de puntos espaciales — evaluación en espacio "
    "condicional (solo celdas con al menos un evento)."
)

tab_pat, tab_k, tab_grilla, tab_mlp, tab_teoria = st.tabs(
    ["Patrones", "Funciones K", "Grilla 4 colores", "Perceptrón Multicapa", "Teoría"]
)


if usar_datos_propios_a and puntos_subidos_a is not None:
    p1 = puntos_subidos_a
else:
    p1 = _patron_simulado(tipo1, n1, seed1, n_grid)

if usar_datos_propios_b and puntos_subidos_b is not None:
    p2 = puntos_subidos_b
else:
    p2 = _patron_condicionado(p1, tipo2, n2, n_coincidentes, seed2, n_grid)

resumen = dataset.analizar_par(p1, p2, r_ref=r_ref, metodo=metodo_k)


# ---------------------------------------------------------------------------
# Tab 1: patrones
# ---------------------------------------------------------------------------
with tab_pat:
    if usar_datos_propios_a and puntos_subidos_a is not None:
        titulo_a = f"Patrón A: datos reales ({len(p1)} puntos)"
    else:
        titulo_a = f"Patrón A: {tipo1}"
    if usar_datos_propios_b and puntos_subidos_b is not None:
        titulo_b = f"Patrón B: datos reales ({len(p2)} puntos)"
    else:
        titulo_b = f"Patrón B: {tipo2}"
    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(viz.fig_patron(p1, "#1565C0", titulo_a, n_grid))
    with c2:
        st.pyplot(viz.fig_patron(p2, "#F9A825", titulo_b, n_grid))
    with c3:
        st.pyplot(viz.fig_grilla_colores(p1, p2, n_grid, "Superposición"))

    n_coinc_real = contar_coincidencias(p1, p2)
    st.subheader("Clasificaciones K de Ripley")
    m1, m2, m3 = st.columns(3)
    m1.metric("Patrón A", resumen.cls1.upper(), f"K={resumen.k1_val:.4f}")
    m2.metric("Patrón B", resumen.cls2.upper(), f"K={resumen.k2_val:.4f}")
    m3.metric("Coocurrencia (K12)", resumen.clase_biv.upper(),
              f"K12={resumen.k12_val:.4f}")
    st.info(
        f"Celdas coincidentes reales (A y B): **{n_coinc_real}** "
        f"(slider = {n_coincidentes}). K teórica CSR en r={r_ref}: "
        f"{np.pi * r_ref**2:.4f}"
    )


# ---------------------------------------------------------------------------
# Tab 2: funciones K
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
        st.pyplot(viz.fig_curva_k(resumen.k1_curva, "K A", "#1565C0",
                                  envs["k1"], r_ref))
    with c2:
        st.pyplot(viz.fig_curva_k(resumen.k2_curva, "K B", "#F9A825",
                                  envs["k2"], r_ref))
    with c3:
        st.pyplot(viz.fig_curva_k(resumen.k12_curva, "K cruzada (K12)",
                                  "#2E7D32", envs["k12"], r_ref))
    st.markdown(
        "- **K > π r²** → agrupamiento / atracción.\n"
        "- **K ≈ π r²** → aleatorio / independencia.\n"
        "- **K < π r²** → dispersión / repulsión."
    )


# ---------------------------------------------------------------------------
# Tab 3: grilla 4 colores
# ---------------------------------------------------------------------------
with tab_grilla:
    st.pyplot(viz.fig_grilla_colores(p1, p2, n_grid, "Grilla — 4 colores"))
    estado = viz.matriz_estados(p1, p2, n_grid)
    n_blanca = int(np.sum(estado == 0))
    n_azul = int(np.sum(estado == 1))
    n_amar = int(np.sum(estado == 2))
    n_verde = int(np.sum(estado == 3))
    cc = st.columns(4)
    cc[0].metric("Blancas (excluidas)", n_blanca)
    cc[1].metric("Solo A (azul)", n_azul)
    cc[2].metric("Solo B (amarillo)", n_amar)
    cc[3].metric("Ambos (verde)", n_verde)
    st.caption(
        "Las celdas blancas se excluyen del cómputo del modelo; se muestran "
        "solo como información descriptiva para el productor."
    )


# ---------------------------------------------------------------------------
# Tab 4: perceptrón multicapa
# ---------------------------------------------------------------------------
with tab_mlp:
    st.write(
        "Entrena el MLP sobre simulaciones y evalúa **solo en celdas "
        "coloreadas**. Genera matrices 3x3 y 2x2."
    )
    p1_fijo_mlp = p1 if (usar_datos_propios_a and puntos_subidos_a is not None) else None
    p2_fijo_mlp = p2 if (usar_datos_propios_b and puntos_subidos_b is not None) else None
    if p1_fijo_mlp is not None and p2_fijo_mlp is not None:
        st.info(
            f"Se usarán ambos patrones cargados como par **fijo** "
            f"({len(p1_fijo_mlp)} puntos A, {len(p2_fijo_mlp)} puntos B) en "
            "todas las simulaciones de entrenamiento. Sin variabilidad entre "
            "simulaciones, el modelo evaluará repetidamente el mismo par real."
        )
    elif p1_fijo_mlp is not None:
        st.info(
            f"Se usará el patrón A cargado ({len(p1_fijo_mlp)} puntos reales) "
            "como patrón A **fijo** en todas las simulaciones de entrenamiento; "
            "solo el patrón B variará."
        )
    elif p2_fijo_mlp is not None:
        st.info(
            f"Se usará el patrón B cargado ({len(p2_fijo_mlp)} puntos reales) "
            "como patrón B **fijo** en todas las simulaciones de entrenamiento; "
            "solo el patrón A variará."
        )
    if st.button("Entrenar y evaluar", type="primary"):
        with st.spinner("Simulando datos y entrenando..."):
            try:
                datos = dataset.simular_dataset(
                    n_sim=n_sim, n_grid=n_grid, n_pts_min=60, n_pts_max=200,
                    r_ref=r_ref, metodo=metodo_k, seed=42,
                    p1_fijo=p1_fijo_mlp, p2_fijo=p2_fijo_mlp,
                )
                if datos.empty:
                    st.error("No se pudo generar el dataset de entrenamiento.")
                    st.stop()
                coloreadas = dataset.filtrar_coloreadas(datos)
                if coloreadas["clase_celda"].nunique() < 2:
                    st.warning(
                        "El dataset tiene solo una clase coloreada. El entrenamiento "
                        "continuará, pero las métricas pueden ser limitadas."
                    )
                fracs = (pct_train / 100, (1 - pct_train / 100) / 2,
                         (1 - pct_train / 100) / 2)
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
            except Exception as exc:  # noqa: BLE001
                st.error(f"Error durante el entrenamiento: {exc}")

    if "rep" in st.session_state:
        rep = st.session_state["rep"]
        info = st.session_state["datos_info"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy (condicional)", f"{rep.accuracy*100:.1f}%")
        c2.metric("Macro F1", f"{rep.macro['f1']:.3f}")
        c3.metric("Celdas blancas (excluidas)", rep.n_blancas)

        st.subheader("Métricas por clase (3 clases)")
        st.dataframe(rep.por_clase.round(3))

        cc1, cc2 = st.columns(2)
        with cc1:
            st.pyplot(viz.fig_matriz_confusion(rep.matriz_3x3, "Matriz 3x3"))
        with cc2:
            st.pyplot(viz.fig_matriz_confusion(rep.matriz_2x2, "Matriz 2x2"))

        if rep.auc_ovr:
            st.write("**AUC-ROC (uno-contra-todos):**",
                     {k: round(v, 3) for k, v in rep.auc_ovr.items()})

        st.pyplot(viz.fig_curvas_aprendizaje(
            st.session_state["res"], "Curvas de aprendizaje"))
        st.caption(
            f"Esquema {info['esquema']} — train coloreadas: {info['n_train']}, "
            f"test coloreadas: {info['n_test']}."
        )


# ---------------------------------------------------------------------------
# Tab 5: teoría
# ---------------------------------------------------------------------------
with tab_teoria:
    st.markdown(
        r"""
### K de Ripley univariada
$$K(r) = \lambda^{-1} \, E[\text{número de puntos en un disco de radio } r]$$

Para CSR (Poisson homogéneo): $K_{teo}(r) = \pi r^2$.

- $K(r) > \pi r^2$: patrón **agrupado** (más puntos cerca de lo esperado al azar).
- $K(r) \approx \pi r^2$: patrón **aleatorio** (CSR).
- $K(r) < \pi r^2$: patrón **disperso** (menos puntos cerca de lo esperado al azar).

### K cruzada (bivariada)
Mide la asociación espacial entre dos tipos de puntos, A y B: cuenta, alrededor
de cada punto de A, cuántos puntos de B caen dentro de un disco de radio $r$,
corregido por borde y normalizado por las intensidades de ambos patrones.

$$K_{12}(r) = \frac{1}{\lambda_2} \, E[\text{número de puntos de B a distancia} \leq r \text{ de un punto de A}]$$

Bajo independencia espacial entre A y B (hipótesis nula), $K_{12}(r) = \pi r^2$,
igual que en el caso univariado. Comparar la curva observada contra esa
referencia teórica (o contra una envolvente de simulación bajo independencia)
permite clasificar la coocurrencia:

- $K_{12}(r) > \pi r^2$: **atracción** — los puntos de A y B tienden a coocurrir
  (coocurrencia positiva, más cercanía mutua de lo esperado al azar).
- $K_{12}(r) \approx \pi r^2$: **sin coocurrencia** — A y B se distribuyen de
  forma espacialmente independiente entre sí.
- $K_{12}(r) < \pi r^2$: **repulsión** — A y B tienden a evitarse mutuamente
  (exclusión espacial, menos cercanía mutua de lo esperado al azar).

### Corrección metodológica (evaluación condicional)
Las celdas blancas (sin A ni B) se **excluyen** del cómputo de accuracy y de
las matrices de confusión. Incluirlas inflaba artificialmente la precisión
porque la clase mayoritaria era "vacío". Se reportan aparte porque son
informativas para el productor, pero no deben usarse para medir el modelo.
"""
    )

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
# Barra lateral: controles
# ---------------------------------------------------------------------------
st.sidebar.title("Controles")

st.sidebar.header("Grilla")
n_grid = st.sidebar.slider("Tamaño de grilla (N x N)", 10, 50, 30, 5)

st.sidebar.header("Patrón A (azul)")
tipo1 = st.sidebar.selectbox("Tipo A", ["agrupado", "disperso", "aleatorio"], 0)
n1 = st.sidebar.slider("Puntos A", 1, 500, 120, 1)
seed1 = st.sidebar.number_input("Semilla A", 1, 999999, 101)

st.sidebar.header("Patrón B (amarillo)")
tipo2 = st.sidebar.selectbox("Tipo B", ["agrupado", "disperso", "aleatorio"], 1)
n2 = st.sidebar.slider("Puntos B", 1, 500, 120, 1)
max_coinc = int(min(n1, n2))
n_coincidentes = st.sidebar.slider(
    "Celdas coincidentes B con A", 0, max_coinc, min(30, max_coinc), 1
)
seed2 = st.sidebar.number_input("Semilla B", 1, 999999, 202)

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


@st.cache_data(show_spinner=False)
def _generar(tipo1, n1, seed1, tipo2, n2, n_coincidentes, seed2, n_grid):
    p1 = generar_patron(tipo1, n1, seed1, n_grid=n_grid)
    p2 = generar_patron_condicionado(
        p1, tipo2, n2, n_coincidentes, seed2, n_grid=n_grid
    )
    return p1, p2


p1, p2 = _generar(tipo1, n1, seed1, tipo2, n2, n_coincidentes, seed2, n_grid)
resumen = dataset.analizar_par(p1, p2, r_ref=r_ref, metodo=metodo_k)


# ---------------------------------------------------------------------------
# Tab 1: patrones
# ---------------------------------------------------------------------------
with tab_pat:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(viz.fig_patron(p1, "#1565C0", f"Patrón A: {tipo1}", n_grid))
    with c2:
        st.pyplot(viz.fig_patron(p2, "#F9A825", f"Patrón B: {tipo2}", n_grid))
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
    if st.button("Entrenar y evaluar", type="primary"):
        with st.spinner("Simulando datos y entrenando..."):
            datos = dataset.simular_dataset(
                n_sim=n_sim, n_grid=n_grid, n_pts_min=60, n_pts_max=200,
                r_ref=r_ref, metodo=metodo_k, seed=42,
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

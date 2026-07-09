# K de Ripley cruzada y Perceptrón Multicapa — versión Python

Reimplementación en **Python** del proyecto original en R (`lpkripleydibujo.R`)
que simula dos patrones de puntos espaciales, calcula la **K de Ripley**
univariada y bivariada (cruzada) y entrena un **perceptrón multicapa (MLP)**
para clasificar la coocurrencia celda a celda en tres categorías:
`atraccion`, `repulsion`, `sin_coocurrencia`.

El proyecto corrige el error metodológico señalado por el profesor y eleva el
rigor del análisis al nivel del taller de posgrado de referencia
(`Taller_Verticillium.pdf`): particiones train/val/test, comparación de
arquitecturas, métricas completas, importancia de variables, regresión
logística como línea base, análisis de dependencia espacial (Moran/LISA) y una
sección de propuesta propia.

---

## Qué corrige respecto al script R original

| Problema en `lpkripleydibujo.R` | Corrección en esta versión |
|---|---|
| **La precisión se calculaba sobre las 900 celdas, incluidas las blancas** (sin evento), inflando el accuracy porque la clase mayoritaria era "vacío". | Todas las métricas (accuracy, precisión/recall/F1, matrices de confusión, ROC-AUC) se calculan **solo sobre celdas coloreadas** (espacio condicional). Las blancas se reportan **aparte**, como información descriptiva. Ver `dataset.filtrar_coloreadas`. |
| Celdas con un solo patrón se etiquetaban como `sin_coocurrencia`. | Se etiquetan como **`repulsion`** (evento único), siguiendo al profesor (min. 35: "cuando aparece solo un color… esa sería la categoría repulsiva"). |
| Puntos limitados a 150; grilla fija 30×30. | Puntos **1–500**; grilla **N×N parametrizable** (default 30×30). |
| B se generaba independiente de A → casi todo caía en "sin_coocurrencia". | B se genera **condicionado** a A con un parámetro de *celdas coincidentes* (coocurrencia real controlada). |
| Clasificación solo por umbral fijo `π r²·1.15/0.85`. | Se implementan **dos métodos**: umbral fijo **y** envolventes de simulación Monte Carlo (CSR / independencia), y se comparan. |
| K de Ripley vía `spatstat`. | K **implementada directamente** (corrección isotrópica de borde), validada contra `π r²` bajo CSR (error < 1 %). |
| `set.seed`. | Reproducibilidad con `numpy.random.default_rng(seed)`. |
| Solo una arquitectura (`nnet`), sin val ni CV. | 70/15/15 vs 80/20, k-fold, comparación de arquitecturas y de funciones de pérdida, importancia de variables, logística, Moran/LISA, RF/SVM, ruido, PCA/t-SNE. |

---

## Estructura del proyecto

```
.
├── coocurrencia/            # Paquete modular (funciones documentadas)
│   ├── patterns.py          # Generación de patrones A y B condicionado
│   ├── ripley.py            # K univariada/cruzada (iso) + envolventes CSR
│   ├── dataset.py           # Dataset a nivel de celda (espacio condicional)
│   ├── metrics.py           # Matrices 3x3/2x2, métricas condicionadas, ROC-AUC
│   ├── models.py            # MLP, particiones, k-fold, curvas, CE vs MSE
│   ├── logistic.py          # Regresión logística binaria/multinomial + selección
│   ├── spatial.py           # Índice de Moran global y local (LISA)
│   ├── importance.py        # Permutation importance y sensibilidad
│   ├── proposals.py         # RF/SVM, robustez a ruido, PCA/t-SNE, regularización
│   └── viz.py               # Visualización (grillas 4 colores, K, matrices…)
├── app/
│   └── streamlit_app.py     # Interfaz interactiva (equivalente a la app Shiny)
├── report/
│   ├── run_pipeline.py      # Ejecuta todo el pipeline y genera figuras/tablas
│   ├── figuras/             # Figuras generadas (PNG)
│   └── tablas/              # Tablas generadas (CSV/JSON/MD)
├── REPORTE.md               # Informe con la estructura del taller de referencia
├── requirements.txt
└── README.md
```

---

## Instalación

Requiere Python 3.11+ (probado en 3.14).

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## Cómo correr todo

### 1) Pipeline completo (figuras + tablas del reporte)

```bash
python report/run_pipeline.py
```

Genera todas las figuras en `report/figuras/`, las tablas en `report/tablas/`
y un resumen en `report/tablas/resultados.json`. Tarda unos minutos
(simulación + comparación de arquitecturas + envolventes + análisis espacial).

### 2) Interfaz interactiva (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Controles: tipo de patrón A/B, número de puntos (1–500), celdas coincidentes,
tamaño de grilla, arquitectura y tasa de aprendizaje del MLP, % de
entrenamiento, radio de referencia y método de etiquetado (umbral/envolvente).
Muestra los patrones, la grilla de 4 colores (blanco/azul/amarillo/verde), las
curvas de K con envolventes, las matrices de confusión 3×3 y 2×2 y la tabla de
métricas condicionadas.

### 3) Uso como librería

```python
from coocurrencia import dataset, models, metrics

datos = dataset.simular_dataset(n_sim=150, n_grid=30, metodo="umbral", seed=42)
part = models.particion_por_sim(datos, (0.70, 0.15, 0.15), seed=42)
res = models.entrenar_mlp(part.train, part.val, hidden_layer_sizes=(16, 8))
rep = metrics.evaluar(part.test["clase_celda"].to_numpy(),
                      res.predecir(part.test),
                      res.predecir_proba(part.test))
print(rep.resumen())
```

---

## Reproducibilidad

Todas las simulaciones usan semillas explícitas (`numpy.random.default_rng`).
Con las semillas por defecto del pipeline (`SEED = 42`), los resultados son
reproducibles ejecución a ejecución.

## Colores de la grilla

- **Blanco**: ninguna ocupación (excluida de métricas del modelo).
- **Azul**: solo patrón A.
- **Amarillo**: solo patrón B.
- **Verde**: ambos (coocurrencia física).

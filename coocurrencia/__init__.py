"""Paquete ``coocurrencia``.

Reimplementación en Python del proyecto original en R (``lpkripleydibujo.R``) que:

1. Simula dos patrones de puntos espaciales en una grilla regular NxN.
2. Calcula la K de Ripley univariada (por patrón) y bivariada/cruzada (K12).
3. Construye un conjunto de datos a nivel de celda para clasificar la
   coocurrencia en tres categorías (atracción / repulsión / sin_coocurrencia).
4. Entrena y compara redes neuronales (MLP) y modelos base (regresión
   logística, Random Forest, SVM) con rigor de taller de posgrado.

Corrección metodológica central respecto al script R:
    Las métricas (accuracy, matrices de confusión, precisión/recall/F1) se
    calculan **solo sobre celdas coloreadas** (con al menos un evento). Las
    celdas blancas se excluyen del cómputo del modelo y se reportan aparte
    como información descriptiva, evitando así inflar artificialmente la
    precisión con la clase mayoritaria "vacío".

Módulos principales
-------------------
- :mod:`coocurrencia.patterns`   Generación de patrones A y B condicionado.
- :mod:`coocurrencia.ripley`     K de Ripley (iso) y envolventes de simulación.
- :mod:`coocurrencia.dataset`    Dataset a nivel de celda (espacio condicional).
- :mod:`coocurrencia.metrics`    Métricas y matrices de confusión condicionadas.
- :mod:`coocurrencia.models`     MLP, comparación de arquitecturas y particiones.
- :mod:`coocurrencia.logistic`   Regresión logística multinomial/binaria.
- :mod:`coocurrencia.spatial`    Índice de Moran global y local (LISA).
- :mod:`coocurrencia.importance` Importancia de variables (permutación/sensibilidad).
- :mod:`coocurrencia.proposals`  Extensiones propias (RF/SVM, ruido, PCA/t-SNE).
- :mod:`coocurrencia.viz`        Utilidades de visualización.
"""

from __future__ import annotations

__version__ = "1.0.0"

CLASES_3 = ["atraccion", "repulsion", "sin_coocurrencia"]
CLASES_2 = ["coocurrente", "no_coocurrente"]
CLASES_UNIV = ["agrupado", "aleatorio", "disperso"]

# Colores usados de forma consistente en toda la app y el reporte.
COLOR_SOLO_A = "#1565C0"   # azul  -> solo patrón A
COLOR_SOLO_B = "#F9A825"   # amarillo -> solo patrón B
COLOR_AMBOS = "#2E7D32"    # verde -> ambos (coocurrencia)
COLOR_BLANCA = "#FFFFFF"   # blanco -> ninguno (excluida de métricas)

__all__ = [
    "CLASES_3",
    "CLASES_2",
    "CLASES_UNIV",
    "COLOR_SOLO_A",
    "COLOR_SOLO_B",
    "COLOR_AMBOS",
    "COLOR_BLANCA",
]

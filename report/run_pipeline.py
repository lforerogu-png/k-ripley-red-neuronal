"""Pipeline completo del proyecto: genera todas las figuras y tablas.

Ejecuta de punta a punta el análisis con el rigor del taller de referencia y
guarda las salidas en ``report/figuras/`` y ``report/tablas/``:

1. Datos: simulación de patrones y construcción del dataset condicional.
2. Comparación de métodos de etiquetado (umbral vs envolvente).
3. Particiones 70/15/15 vs 80/20 y validación cruzada k-fold.
4. Comparación sistemática de arquitecturas de MLP + curvas de aprendizaje.
5. Comparación de funciones de pérdida (entropía cruzada vs MSE).
6. Métricas completas 3 clases (3x3, ROC-AUC) y binario (2x2).
7. Regresión logística (binaria y multinomial) + selección de variables.
8. Importancia de variables (permutación) vs coeficientes logísticos.
9. Análisis espacial: Moran global y LISA.
10. Propuestas propias: RF/SVM, robustez ante ruido, PCA/t-SNE, regularización.

Uso:
    python report/run_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coocurrencia import (  # noqa: E402
    CLASES_3, dataset, importance, logistic, metrics, models, proposals,
    spatial, viz,
)
from coocurrencia.patterns import generar_patron, generar_patron_condicionado  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")
TAB = os.path.join(AQUI, "tablas")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

SEED = 42
N_GRID = 30
N_SIM = 150

resultados = {}
tablas_md = []


def guardar_fig(fig, nombre):
    ruta = os.path.join(FIG, nombre)
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    print(f"  figura -> {os.path.relpath(ruta, AQUI)}")


def guardar_tabla(df, nombre, titulo):
    ruta = os.path.join(TAB, nombre)
    df.to_csv(ruta, index=False)
    tablas_md.append(f"### {titulo}\n\n{df.to_markdown(index=False)}\n")
    print(f"  tabla  -> {os.path.relpath(ruta, AQUI)}")


def main():
    print("=" * 70)
    print("PIPELINE: K de Ripley + Perceptrón Multicapa (espacio condicional)")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 0. Ejemplo ilustrativo de un par de patrones (figuras de portada)
    # -----------------------------------------------------------------
    print("\n[0] Par de patrones de ejemplo...")
    p1 = generar_patron("agrupado", 150, 101, n_grid=N_GRID)
    p2 = generar_patron_condicionado(p1, "agrupado", 150, 60, 202, n_grid=N_GRID)
    res_par = dataset.analizar_par(p1, p2, r_ref=0.2, metodo="umbral")
    guardar_fig(viz.fig_grilla_colores(p1, p2, N_GRID), "00_grilla_4colores.png")
    guardar_fig(viz.fig_curva_k(res_par.k12_curva, "K cruzada (K12)",
                                "#2E7D32", None, 0.2), "00_k12.png")

    # -----------------------------------------------------------------
    # 1. Dataset principal (umbral)
    # -----------------------------------------------------------------
    print(f"\n[1] Simulando dataset principal (n_sim={N_SIM}, grilla {N_GRID})...")
    datos = dataset.simular_dataset(
        n_sim=N_SIM, n_grid=N_GRID, n_pts_min=60, n_pts_max=200,
        metodo="umbral", seed=SEED, verbose=True,
    )
    col = dataset.filtrar_coloreadas(datos)
    n_blancas = int(datos["celda_blanca"].sum())
    dist3 = col["clase_celda"].value_counts()
    resultados["n_total_celdas"] = int(len(datos))
    resultados["n_blancas"] = n_blancas
    resultados["n_coloreadas"] = int(len(col))
    resultados["pct_blancas"] = round(100 * n_blancas / len(datos), 1)
    resultados["distribucion_clases"] = dist3.to_dict()
    print(f"  celdas totales={len(datos)}  blancas={n_blancas} "
          f"({resultados['pct_blancas']}%)  coloreadas={len(col)}")

    df_dist = dist3.reset_index()
    df_dist.columns = ["clase", "n_celdas"]
    guardar_tabla(df_dist, "01_distribucion_clases.csv",
                  "Distribución de clases (celdas coloreadas)")

    # -----------------------------------------------------------------
    # 2. Comparación de métodos de etiquetado (umbral vs envolvente)
    # -----------------------------------------------------------------
    print("\n[2] Comparando etiquetado umbral vs envolvente (subconjunto)...")
    acuerdo = comparar_etiquetado(n_pares=40)
    resultados["acuerdo_umbral_envolvente"] = acuerdo
    print(f"  acuerdo clase bivariada umbral vs envolvente: {acuerdo}%")

    # -----------------------------------------------------------------
    # 3. Particiones 70/15/15 y 80/20
    # -----------------------------------------------------------------
    print("\n[3] Particiones 70/15/15 vs 80/20...")
    part3 = models.particion_por_sim(datos, (0.70, 0.15, 0.15), seed=SEED)
    part2 = models.particion_por_sim(datos, (0.80, 0.20), seed=SEED)
    filas_part = []
    for nombre, part in [("70/15/15", part3), ("80/20", part2)]:
        res = models.entrenar_mlp(part.train, part.val, hidden_layer_sizes=(16, 8),
                                  learning_rate_init=0.01, max_epocas=200)
        acc = metrics.accuracy_condicional(part.test, res)
        filas_part.append({"esquema": nombre,
                           "n_train": len(part.train), "n_test": len(part.test),
                           "accuracy_test": round(acc, 4)})
    guardar_tabla(pd.DataFrame(filas_part), "03_particiones.csv",
                  "Comparación de esquemas de partición")

    # -----------------------------------------------------------------
    # 4. Validación cruzada k-fold
    # -----------------------------------------------------------------
    print("\n[4] Validación cruzada k-fold (agrupada por simulación)...")
    cv = models.validacion_cruzada(datos, k=5, hidden_layer_sizes=(16, 8),
                                   learning_rate_init=0.01, max_epocas=150)
    resultados["cv_media"] = round(float(cv.attrs["media"]), 4)
    resultados["cv_std"] = round(float(cv.attrs["std"]), 4)
    guardar_tabla(cv.round(4), "04_kfold.csv",
                  f"Validación cruzada 5-fold (media={resultados['cv_media']})")

    # -----------------------------------------------------------------
    # 5. Comparación de arquitecturas
    # -----------------------------------------------------------------
    print("\n[5] Comparando arquitecturas (capas 1/2/3 x lr 0.001/0.01/0.1)...")
    tabla_arq, res_arq = models.comparar_arquitecturas(
        part3, max_epocas=150, seed=SEED)
    guardar_tabla(tabla_arq.round(4), "05_arquitecturas.csv",
                  "Comparación de arquitecturas de MLP")
    mejor = tabla_arq.iloc[0]["config"]
    resultados["mejor_arquitectura"] = mejor
    print(f"  mejor configuración: {mejor}")

    # Curvas de aprendizaje del mejor y de uno con sobreajuste (lr alto).
    guardar_fig(viz.fig_curvas_aprendizaje(res_arq[mejor], f"Mejor: {mejor}"),
                "05_curvas_mejor.png")
    lr_alto = [k for k in res_arq if k.endswith("lr0.1")]
    if lr_alto:
        guardar_fig(viz.fig_curvas_aprendizaje(res_arq[lr_alto[0]],
                    f"lr alto: {lr_alto[0]}"), "05_curvas_lr_alto.png")

    # -----------------------------------------------------------------
    # 6. Función de pérdida: entropía cruzada vs MSE
    # -----------------------------------------------------------------
    print("\n[6] Comparando funciones de pérdida (CE vs MSE)...")
    cmp_loss = models.comparar_perdidas(part3, hidden_layer_sizes=(16, 8),
                                        learning_rate_init=0.01, max_epocas=250)
    guardar_tabla(cmp_loss.round(4), "06_perdidas.csv",
                  "Entropía cruzada vs MSE")

    # -----------------------------------------------------------------
    # 7. Métricas completas del mejor modelo (3 clases)
    # -----------------------------------------------------------------
    print("\n[7] Métricas completas del modelo final (3 clases)...")
    hidden_mejor = res_arq[mejor].config["hidden_layer_sizes"]
    lr_mejor = res_arq[mejor].config["learning_rate_init"]
    res_final = models.entrenar_mlp(part3.train, part3.val,
                                    hidden_layer_sizes=hidden_mejor,
                                    learning_rate_init=lr_mejor, max_epocas=300)
    pred = res_final.predecir(part3.test)
    proba = res_final.predecir_proba(part3.test)
    n_bl_test = int(datos[datos["sim_id"].isin(
        part3.test["sim_id"].unique())]["celda_blanca"].sum())
    rep = metrics.evaluar(part3.test["clase_celda"].to_numpy(), pred, proba,
                          clases=CLASES_3, n_blancas=n_bl_test)
    resultados["accuracy_3clases"] = round(rep.accuracy, 4)
    resultados["macro_f1_3clases"] = round(rep.macro["f1"], 4)
    resultados["auc_ovr"] = {k: round(v, 3) for k, v in (rep.auc_ovr or {}).items()}
    print("  " + rep.resumen().replace("\n", "\n  "))
    guardar_tabla(rep.por_clase.round(3).reset_index().rename(
        columns={"index": "clase"}), "07_metricas_por_clase.csv",
        "Métricas por clase (3 clases, test condicional)")
    guardar_fig(viz.fig_matriz_confusion(rep.matriz_3x3, "Matriz 3x3 (condicional)"),
                "07_matriz_3x3.png")
    guardar_fig(viz.fig_matriz_confusion(rep.matriz_2x2, "Matriz 2x2 (condicional)"),
                "07_matriz_2x2.png")

    # -----------------------------------------------------------------
    # 8. Modelo binario
    # -----------------------------------------------------------------
    print("\n[8] Modelo binario (coocurrente vs no_coocurrente)...")
    res_bin = models.entrenar_mlp(part3.train, part3.val, target="clase_binaria",
                                  hidden_layer_sizes=hidden_mejor,
                                  learning_rate_init=lr_mejor, max_epocas=250)
    pred_bin = res_bin.predecir(part3.test)
    from coocurrencia import CLASES_2
    rep_bin = metrics.evaluar(part3.test["clase_binaria"].to_numpy(), pred_bin,
                              clases=CLASES_2, n_blancas=n_bl_test)
    det = metrics.metricas_binarias_detalle(
        part3.test["clase_binaria"].to_numpy(), pred_bin)
    resultados["accuracy_binario"] = round(rep_bin.accuracy, 4)
    resultados["metricas_binario"] = {k: round(v, 4) for k, v in det.items()}
    guardar_tabla(pd.DataFrame([det]).round(4), "08_binario.csv",
                  "Métricas modelo binario (test condicional)")

    # -----------------------------------------------------------------
    # 9. Regresión logística + selección de variables
    # -----------------------------------------------------------------
    print("\n[9] Regresión logística (binaria y multinomial)...")
    lb = logistic.logistica_binaria(part3.train, part3.test)
    guardar_tabla(lb["coeficientes"].round(4), "09_logit_binaria.csv",
                  "Regresión logística binaria (coeficientes)")
    resultados["logit_binaria_auc"] = round(float(lb["auc"]), 4)
    resultados["logit_binaria_separacion"] = bool(lb["separacion"])

    sel_aic = logistic.seleccion_backward_aic(part3.train)
    resultados["seleccion_aic"] = sel_aic["seleccion_final"]
    lasso = logistic.seleccion_lasso(part3.train)
    guardar_tabla(lasso.round(4), "09_lasso.csv",
                  "Selección de variables por Lasso (L1)")

    mm = logistic.logistica_multinomial(part3.train, part3.test)
    resultados["logit_multinomial_acc"] = round(mm["reporte"].accuracy, 4)

    # -----------------------------------------------------------------
    # 10. Importancia de variables
    # -----------------------------------------------------------------
    print("\n[10] Importancia de variables (permutación vs logística)...")
    perm = importance.permutation_importance(res_final, part3.test,
                                             n_repeticiones=20)
    guardar_tabla(perm.round(4), "10_permutation_importance.csv",
                  "Importancia por permutación (MLP)")
    comp = importance.comparar_importancias(perm, lb["coeficientes"])
    guardar_tabla(comp.round(4), "10_importancia_comparada.csv",
                  "Ranking permutación (MLP) vs |coef| logístico")

    # -----------------------------------------------------------------
    # 11. Análisis espacial (Moran + LISA)
    # -----------------------------------------------------------------
    # Se compara un patrón agrupado (debería mostrar coherencia espacial) con
    # uno aleatorio (sin dependencia), para ilustrar el estadístico de Moran.
    print("\n[11] Análisis espacial (Moran global + LISA)...")
    W = spatial.matriz_pesos_grilla(N_GRID, "queen")
    import collections

    def moran_de_par(t1, t2, nc, tag, guardar=False):
        pa = generar_patron(t1, 180, 777, n_grid=N_GRID)
        pb = generar_patron_condicionado(pa, t2, 180, nc, 888, n_grid=N_GRID)
        est = viz.matriz_estados(pa, pb, N_GRID)  # 0..3 en (row-1, col-1)
        valores = est.ravel().astype(float)       # orden fila-mayor = W
        mg = spatial.moran_global(valores, W, n_perm=499)
        li = spatial.moran_local(valores, W, n_perm=199)
        if guardar:
            guardar_fig(viz.fig_lisa(li.cluster, N_GRID,
                        f"Clústeres LISA ({tag})"), "11_lisa.png")
        return mg, dict(collections.Counter(li.cluster))

    mg_agr, lisa_agr = moran_de_par("agrupado", "agrupado", 90, "agrupado",
                                    guardar=True)
    mg_ale, _ = moran_de_par("aleatorio", "aleatorio", 20, "aleatorio")
    resultados["moran_agrupado_I"] = round(mg_agr.I, 4)
    resultados["moran_agrupado_p"] = round(mg_agr.p_valor, 4)
    resultados["moran_agrupado_interp"] = mg_agr.interpreta()
    resultados["moran_aleatorio_I"] = round(mg_ale.I, 4)
    resultados["moran_aleatorio_p"] = round(mg_ale.p_valor, 4)
    resultados["lisa_clusters_agrupado"] = lisa_agr
    guardar_tabla(pd.DataFrame([
        {"patron": "agrupado", "moran_I": round(mg_agr.I, 4),
         "p_valor": round(mg_agr.p_valor, 4), "interpretacion": mg_agr.interpreta()},
        {"patron": "aleatorio", "moran_I": round(mg_ale.I, 4),
         "p_valor": round(mg_ale.p_valor, 4), "interpretacion": mg_ale.interpreta()},
    ]), "11_moran.csv", "Índice de Moran global: agrupado vs aleatorio")
    print(f"  agrupado: Moran I={mg_agr.I:.4f} p={mg_agr.p_valor:.4f}")
    print(f"  aleatorio: Moran I={mg_ale.I:.4f} p={mg_ale.p_valor:.4f}")

    # -----------------------------------------------------------------
    # 12. Propuestas propias
    # -----------------------------------------------------------------
    print("\n[12] Propuestas propias (RF/SVM, ruido, PCA/t-SNE, regularización)...")
    alt = proposals.comparar_modelos_alternativos(part3)
    guardar_tabla(alt.round(4), "12_modelos_alternativos.csv",
                  "MLP vs Random Forest vs SVM")
    ruido = proposals.robustez_ante_ruido(part3)
    guardar_tabla(ruido.round(4), "12_robustez_ruido.csv",
                  "Robustez ante ruido en las K")
    reg = proposals.efecto_regularizacion(part3)
    guardar_tabla(reg.round(4), "12_regularizacion.csv",
                  "Efecto de early stopping y L2 (alpha)")
    pca = proposals.reducir_dimension(col, metodo="pca", seed=SEED)
    guardar_fig(viz.fig_dispersion_2d(pca, "PCA de las entradas"),
                "12_pca.png")
    try:
        tsne = proposals.reducir_dimension(col, metodo="tsne", seed=SEED)
        guardar_fig(viz.fig_dispersion_2d(tsne, "t-SNE de las entradas"),
                    "12_tsne.png")
    except Exception as e:
        print(f"  (t-SNE omitido: {e})")

    # -----------------------------------------------------------------
    # Guardar resultados agregados
    # -----------------------------------------------------------------
    with open(os.path.join(TAB, "resultados.json"), "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(TAB, "resultados_tablas.md"), "w", encoding="utf-8") as fh:
        fh.write("# Tablas generadas por el pipeline\n\n")
        fh.write("\n".join(tablas_md))

    print("\n" + "=" * 70)
    print("RESULTADOS CLAVE")
    print("=" * 70)
    for k, v in resultados.items():
        print(f"  {k}: {v}")
    print(f"\nFiguras en: {FIG}\nTablas en:  {TAB}")


def comparar_etiquetado(n_pares: int = 40) -> float:
    """Compara la clase bivariada por umbral vs por envolvente en pares comunes."""
    rng = np.random.default_rng(SEED)
    tipos = ("agrupado", "disperso", "aleatorio")
    iguales = 0
    total = 0
    for _ in range(n_pares):
        s = int(rng.integers(1, 10_000_000))
        t1, t2 = tipos[rng.integers(3)], tipos[rng.integers(3)]
        n = int(rng.integers(60, 200))
        p1 = generar_patron(t1, n, s, n_grid=N_GRID)
        if len(p1) == 0:
            continue
        nc = int(rng.integers(0, min(len(p1), n) + 1))
        p2 = generar_patron_condicionado(p1, t2, n, nc, s + 500, n_grid=N_GRID)
        if len(p2) == 0:
            continue
        ru = dataset.analizar_par(p1, p2, metodo="umbral")
        re = dataset.analizar_par(p1, p2, metodo="envolvente", n_sim_env=39)
        total += 1
        iguales += int(ru.clase_biv == re.clase_biv)
    return round(100 * iguales / total, 1) if total else float("nan")


if __name__ == "__main__":
    main()

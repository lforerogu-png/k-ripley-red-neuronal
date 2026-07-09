# ============================================================
# APP SHINY: K de Ripley Cruzada y Perceptrón Multicapa
# Coocurrencia en Patrones de Puntos Espaciales
# Evaluación en espacio condicional (solo celdas con eventos)
# ============================================================

library(shiny)
library(spatstat.geom)
library(spatstat.explore)
library(nnet)
library(ggplot2)
library(dplyr)
library(caret)

# ============================================================
# FUNCIONES AUXILIARES — GENERACIÓN DE PATRONES
# ============================================================

generar_patron <- function(tipo, n_puntos = 60, seed = NULL,
                           celdas_ocupadas = NULL) {
  if (!is.null(seed)) set.seed(seed)
  celdas <- expand.grid(col = 1:30, row = 1:30)

  if (!is.null(celdas_ocupadas) && nrow(celdas_ocupadas) > 0) {
    ocupadas <- paste(celdas_ocupadas$col, celdas_ocupadas$row)
    celdas   <- celdas[!paste(celdas$col, celdas$row) %in% ocupadas, , drop = FALSE]
  }

  if (tipo == "agrupado") {
    n_clusters <- 4
    centros <- celdas[sample(nrow(celdas), min(n_clusters, nrow(celdas))), , drop = FALSE]
    puntos <- do.call(rbind, lapply(seq_len(nrow(centros)), function(i) {
      n_i <- round(n_puntos / nrow(centros))
      cx  <- centros$col[i]; cy <- centros$row[i]
      cand <- celdas[abs(celdas$col - cx) <= 6 &
                       abs(celdas$row - cy) <= 6, , drop = FALSE]
      if (nrow(cand) < n_i) cand <- celdas
      cand[sample(nrow(cand), min(n_i, nrow(cand))), , drop = FALSE]
    }))
    puntos <- puntos[!duplicated(puntos), , drop = FALSE]
    if (nrow(puntos) > n_puntos) puntos <- puntos[seq_len(n_puntos), , drop = FALSE]

  } else if (tipo == "disperso") {
    if (nrow(celdas) == 0) {
      puntos <- celdas_ocupadas[0, , drop = FALSE]
    } else {
      selec    <- celdas[sample(nrow(celdas), 1), , drop = FALSE]
      intentos <- 0
      while (nrow(selec) < n_puntos && intentos < 8000) {
        cand  <- celdas[sample(nrow(celdas), 1), , drop = FALSE]
        dists <- sqrt((selec$col - cand$col)^2 +
                        (selec$row - cand$row)^2)
        ya <- any(selec$col == cand$col & selec$row == cand$row)
        if (min(dists) >= 3 && !ya) selec <- rbind(selec, cand)
        intentos <- intentos + 1
      }
      puntos <- selec
    }

  } else {
    n_sel <- min(n_puntos, nrow(celdas))
    if (n_sel == 0) {
      puntos <- celdas_ocupadas[0, , drop = FALSE]
    } else {
      idx    <- sample(nrow(celdas), n_sel)
      puntos <- celdas[idx, , drop = FALSE]
    }
  }

  if (nrow(puntos) == 0) {
    puntos <- data.frame(col = integer(), row = integer())
  }

  puntos$x <- (puntos$col - 0.5) / 30
  puntos$y <- (puntos$row - 0.5) / 30
  puntos
}

#' Genera el Patrón 2 forzando n_coincidentes celdas compartidas con P1.
generar_patron2_con_coincidencia <- function(p1, tipo2, n_puntos, n_coincidentes,
                                             seed = NULL) {
  n_coincidentes <- max(0L, min(as.integer(n_coincidentes),
                                  nrow(p1), as.integer(n_puntos)))

  if (n_coincidentes > 0 && nrow(p1) > 0) {
    idx_coinc <- sample(nrow(p1), n_coincidentes)
    coinc     <- p1[idx_coinc, c("col", "row"), drop = FALSE]
  } else {
    coinc <- data.frame(col = integer(), row = integer())
  }

  n_restantes <- as.integer(n_puntos) - n_coincidentes
  if (n_restantes > 0) {
    libres <- generar_patron(tipo2, n_restantes, seed, celdas_ocupadas = coinc)
    if (nrow(libres) > 0) {
      p2 <- rbind(coinc, libres[, c("col", "row"), drop = FALSE])
      p2 <- p2[!duplicated(p2), , drop = FALSE]
    } else {
      p2 <- coinc
    }
  } else {
    p2 <- coinc
  }

  p2$x <- (p2$col - 0.5) / 30
  p2$y <- (p2$row - 0.5) / 30
  p2
}

hacer_ppp <- function(p) {
  if (nrow(p) == 0) {
  return(ppp(numeric(0), numeric(0), window = owin(c(0, 1), c(0, 1))))
  }
  ppp(p$x, p$y, window = owin(c(0, 1), c(0, 1)))
}

extraer_K_val <- function(K_obj, r_ref = 0.2) {
  idx <- which.min(abs(K_obj$r - r_ref))
  val <- K_obj$iso[idx]
  if (is.na(val)) pi * r_ref^2 else val
}

clasificar_univ <- function(K_obj, r_ref = 0.2) {
  K_teo <- pi * r_ref^2
  val   <- extraer_K_val(K_obj, r_ref)
  if      (val > K_teo * 1.15) "agrupado"
  else if (val < K_teo * 0.85) "disperso"
  else                          "aleatorio"
}

clasificar_biv <- function(K12, r_ref = 0.2) {
  K_teo <- pi * r_ref^2
  val   <- extraer_K_val(K12, r_ref)
  if      (val > K_teo * 1.15) "atraccion"
  else if (val < K_teo * 0.85) "repulsion"
  else                          "sin_coocurrencia"
}

# ============================================================
# DATOS A NIVEL DE CELDA (espacio condicional)
# ============================================================

etiqueta_celda <- function(ocupa_p1, ocupa_p2, clase_biv) {
  ifelse(
    ocupa_p1 & ocupa_p2,
    clase_biv,
    "sin_coocurrencia"
  )
}

construir_datos_celdas <- function(p1, p2, k1_val, k2_val, k12_val,
                                   cls1, cls2, clase_biv, k_max = NULL) {
  grilla <- expand.grid(col = 1:30, row = 1:30)
  key    <- paste(grilla$col, grilla$row)

  p1_en <- key %in% paste(p1$col, p1$row)
  p2_en <- key %in% paste(p2$col, p2$row)

  grilla$ocupa_p1 <- as.integer(p1_en)
  grilla$ocupa_p2 <- as.integer(p2_en)
  grilla$celda_blanca <- grilla$ocupa_p1 == 0L & grilla$ocupa_p2 == 0L

  cls1_f <- as.numeric(factor(cls1,
                              levels = c("agrupado", "aleatorio", "disperso")))
  cls2_f <- as.numeric(factor(cls2,
                              levels = c("agrupado", "aleatorio", "disperso")))

  grilla$k1_val  <- k1_val
  grilla$k2_val  <- k2_val
  grilla$k12_val <- k12_val
  if (!is.null(k_max) && is.finite(k_max) && k_max > 0) {
    grilla$k1_n  <- k1_val / k_max
    grilla$k2_n  <- k2_val / k_max
    grilla$k12_n <- k12_val / k_max
  }
  grilla$cls1_f <- cls1_f
  grilla$cls2_f <- cls2_f

  grilla$clase <- etiqueta_celda(
    grilla$ocupa_p1 == 1L,
    grilla$ocupa_p2 == 1L,
    clase_biv
  )
  grilla$clase_f <- factor(
    grilla$clase,
    levels = c("atraccion", "sin_coocurrencia", "repulsion")
  )

  grilla
}

simular_datos_celdas <- function(n_sim = 200, n_pts = 60) {
  tipos <- c("agrupado", "disperso", "aleatorio")
  bloques <- vector("list", n_sim)
  ok <- 0L

  for (i in seq_len(n_sim * 4L)) {
    if (ok >= n_sim) break

    t1 <- sample(tipos, 1)
    t2 <- sample(tipos, 1)

    p1 <- tryCatch(generar_patron(t1, n_pts, i),
                   error = function(e) NULL)
    if (is.null(p1) || nrow(p1) == 0) next

    n_coinc <- sample(0:min(nrow(p1), n_pts), 1)
    p2 <- tryCatch(
      generar_patron2_con_coincidencia(p1, t2, n_pts, n_coinc, i + 1e4),
      error = function(e) NULL
    )
    if (is.null(p2) || nrow(p2) == 0) next

    pp1 <- hacer_ppp(p1)
    pp2 <- hacer_ppp(p2)
    K1  <- tryCatch(Kest(pp1, correction = "iso"), error = function(e) NULL)
    K2  <- tryCatch(Kest(pp2, correction = "iso"), error = function(e) NULL)
    ppc <- tryCatch(superimpose(A = pp1, B = pp2), error = function(e) NULL)
    K12 <- tryCatch(
      Kcross(ppc, "A", "B", correction = "iso"),
      error = function(e) NULL
    )
    if (any(sapply(list(K1, K2, K12), is.null))) next

    k1_val  <- extraer_K_val(K1)
    k2_val  <- extraer_K_val(K2)
    k12_val <- extraer_K_val(K12)
    cls1    <- clasificar_univ(K1)
    cls2    <- clasificar_univ(K2)
    clase   <- clasificar_biv(K12)

    ok <- ok + 1L
    bloque <- construir_datos_celdas(p1, p2, k1_val, k2_val, k12_val,
                                     cls1, cls2, clase)
    bloque$sim_id <- ok
    bloques[[ok]] <- bloque
  }

  if (ok == 0L) return(data.frame())

  datos <- bind_rows(bloques[seq_len(ok)])
  k_max <- max(c(datos$k1_val, datos$k2_val, datos$k12_val), na.rm = TRUE) + 1e-6

  datos$k1_n  <- datos$k1_val  / k_max
  datos$k2_n  <- datos$k2_val  / k_max
  datos$k12_n <- datos$k12_val / k_max

  attr(datos, "k_max") <- k_max
  datos
}

filtrar_celdas_coloreadas <- function(datos) {
  datos %>% filter(!celda_blanca)
}

colapsar_clase_binaria <- function(x) {
  factor(
    ifelse(x %in% c("atraccion", "repulsion"), "Coocurrencia", "No_Coocurrencia"),
    levels = c("Coocurrencia", "No_Coocurrencia")
  )
}

matriz_confusion_2x2 <- function(real, pred) {
  real_b <- colapsar_clase_binaria(real)
  pred_b <- colapsar_clase_binaria(pred)
  table(
    Pred = pred_b,
    Real = real_b,
    useNA = "no"
  )
}

calcular_metricas_condicionadas <- function(real, pred) {
  real_b <- colapsar_clase_binaria(real)
  pred_b <- colapsar_clase_binaria(pred)
  cm2    <- table(Pred = pred_b, Real = real_b, useNA = "no")

  lvls <- c("Coocurrencia", "No_Coocurrencia")
  cm <- matrix(0, nrow = 2, ncol = 2,
               dimnames = list(Pred = lvls, Real = lvls))
  for (r in rownames(cm2)) {
    for (c in colnames(cm2)) {
      if (r %in% lvls && c %in% lvls) cm[r, c] <- cm2[r, c]
    }
  }

  TP <- cm["Coocurrencia", "Coocurrencia"]
  TN <- cm["No_Coocurrencia", "No_Coocurrencia"]
  FP <- cm["Coocurrencia", "No_Coocurrencia"]
  FN <- cm["No_Coocurrencia", "Coocurrencia"]
  n  <- sum(cm)

  accuracy    <- if (n > 0) (TP + TN) / n else NA_real_
  sensitivity <- if ((TP + FN) > 0) TP / (TP + FN) else NA_real_
  specificity <- if ((TN + FP) > 0) TN / (TN + FP) else NA_real_

  if (n > 0) {
    cm_caret <- confusionMatrix(pred_b, real_b, positive = "Coocurrencia")
    accuracy    <- as.numeric(cm_caret$overall["Accuracy"])
    sensitivity <- as.numeric(cm_caret$byClass["Sensitivity"])
    specificity <- as.numeric(cm_caret$byClass["Specificity"])
  }

  list(
    accuracy    = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    cm2         = cm
  )
}

# ============================================================
# HELPERS DE GRÁFICAS
# ============================================================

gg_puntos <- function(p, col, tit) {
  grilla_df <- expand.grid(gx = 1:30, gy = 1:30)
  ggplot() +
    geom_tile(data = grilla_df,
              aes(x = gx, y = gy),
              fill = "white", color = "gray85",
              linewidth = 0.25) +
    geom_point(data  = p,
               aes(x = col, y = row),
               color = col, size = 3, alpha = 0.8) +
    coord_equal() +
    scale_x_continuous(limits = c(0.5, 30.5)) +
    scale_y_continuous(limits = c(0.5, 30.5)) +
    labs(title = tit, x = NULL, y = NULL) +
    theme_minimal(base_size = 9) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
}

gg_K <- function(K_obj, col, tit) {
  df     <- as.data.frame(K_obj)
  df$teo <- pi * df$r^2
  ggplot(df, aes(r)) +
    geom_line(aes(y = iso), color = col, linewidth = 1.2) +
    geom_line(aes(y = teo), linetype = "dashed",
              color = "gray40", linewidth = 0.8) +
    labs(title = tit, x = "Radio r", y = "K(r)") +
    theme_minimal(base_size = 9) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
}

gg_grilla <- function(p, col, tit) {
  g     <- expand.grid(gx = 1:30, gy = 1:30)
  g$ocu <- paste(g$gx, g$gy) %in% paste(p$col, p$row)
  ggplot(g, aes(x = gx, y = gy, fill = ocu)) +
    geom_tile(color = "gray80", linewidth = 0.25) +
    scale_fill_manual(
      values = c("TRUE" = col, "FALSE" = "white"),
      labels = c("TRUE" = "Punto", "FALSE" = "Vacio")) +
    coord_equal() +
    labs(title = tit, x = NULL, y = NULL, fill = NULL) +
    theme_minimal(base_size = 8) +
    theme(plot.title       = element_text(face = "bold", hjust = 0.5),
          legend.position  = "bottom",
          panel.grid       = element_blank())
}

# ============================================================
# UI
# ============================================================
ui <- fluidPage(

  tags$head(tags$style(HTML("
    .metric-card {
      background: #ffffff;
      border: 2px solid #1565C0;
      border-radius: 10px;
      padding: 18px 10px;
      text-align: center;
      margin-bottom: 10px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .metric-value {
      font-size: 2.2em;
      font-weight: bold;
      color: #0d47a1;
      line-height: 1.1;
    }
    .metric-label {
      font-size: 0.95em;
      color: #455a64;
      margin-top: 6px;
    }
    .metric-panel {
      background: #e3f2fd;
      border-radius: 10px;
      padding: 16px;
      border: 1px solid #90caf9;
      margin-bottom: 16px;
    }
  "))),

  titlePanel(
    div(
      h2("K de Ripley Cruzada — Coocurrencia en Patrones de Puntos",
         style = "color:#003366;font-weight:bold;"),
      h4("Clasificacion univariada, bivariada y Perceptron Multicapa",
         style = "color:#555;"),
      p(style = "color:#666;font-size:0.95em;",
        "Evaluacion en espacio condicional: solo celdas con al menos un evento espacial.")
    )
  ),

  sidebarLayout(
    sidebarPanel(
      width = 3,

      h4("Patron 1 (Azul)", style = "color:#003399;"),
      selectInput("tipo1", "Tipo:",
                  c("agrupado", "disperso", "aleatorio"), "agrupado"),
      sliderInput("n1", "Puntos:", 30, 150, 60, 10),
      numericInput("seed1", "Semilla:", 101, 1, 9999),

      hr(),
      h4("Patron 2 (Rojo)", style = "color:#990000;"),
      selectInput("tipo2", "Tipo:",
                  c("agrupado", "disperso", "aleatorio"), "disperso"),
      sliderInput("n2", "Puntos:", 30, 150, 60, 10),
      sliderInput("n_coincidentes",
                  "Puntos coincidentes con P1:",
                  min = 0, max = 60, value = 10, step = 1),
      helpText("Forza superposicion exacta en la misma celda de la grilla."),
      numericInput("seed2", "Semilla:", 202, 1, 9999),

      hr(),
      h4("Red Neuronal (nnet)", style = "color:#006600;"),
      sliderInput("n_sim", "Simulaciones entrenamiento:",
                  100, 500, 200, 50),
      sliderInput("pct_train", "% datos entrenamiento:",
                  50, 90, 80, step = 5, post = "%"),
      sliderInput("size_nn", "Neuronas ocultas (size):",
                  min = 2, max = 15, value = 8, step = 1),
      selectInput("decay_nn", "Decay (regularizacion L2):",
                  choices = c("0" = 0, "0.01" = 0.01,
                              "0.1" = 0.1, "0.5" = 0.5),
                  selected = 0.01),
      helpText("Arquitectura justificable segun rnpm_redes.pdf: size y decay controlan capacidad y sobreajuste."),

      hr(),
      actionButton("simular", "Generar Patrones",
                   class = "btn-primary", width = "100%"),
      br(), br(),
      actionButton("entrenar", "Entrenar Red Neuronal",
                   class = "btn-success", width = "100%"),
      br(), br(),
      actionButton("predecir", "Clasificar con la Red",
                   class = "btn-warning", width = "100%")
    ),

    mainPanel(
      width = 9,
      tabsetPanel(

        tabPanel("Patrones de Puntos", br(),
                 fluidRow(
                   column(4,
                          h5("Patron 1 (Azul)",
                             style = "text-align:center;color:#003399;"),
                          plotOutput("plt_p1", height = "270px")),
                   column(4,
                          h5("Patron 2 (Rojo)",
                             style = "text-align:center;color:#990000;"),
                          plotOutput("plt_p2", height = "270px")),
                   column(4,
                          h5("Superposicion", style = "text-align:center;"),
                          plotOutput("plt_super", height = "270px"))
                 ),
                 hr(),
                 div(style = "background:#f0f4ff;padding:12px;border-radius:8px;",
                     h4("Clasificaciones K de Ripley", style = "margin-top:0;"),
                     verbatimTextOutput("resumen_K"))
        ),

        tabPanel("Funciones K", br(),
                 fluidRow(
                   column(4, h5("K Patron 1", style = "text-align:center;"),
                          plotOutput("plt_K1", height = "270px")),
                   column(4, h5("K Patron 2", style = "text-align:center;"),
                          plotOutput("plt_K2", height = "270px")),
                   column(4, h5("K Cruzada", style = "text-align:center;"),
                          plotOutput("plt_K12", height = "270px"))
                 ),
                 hr(),
                 div(style = "background:#fff8e1;padding:12px;border-radius:8px;",
                     h4("Interpretacion", style = "margin-top:0;"),
                     uiOutput("interp_K"))
        ),

        tabPanel("Grilla 30x30", br(),
                 fluidRow(
                   column(6, h5("Patron 1 en grilla", style = "text-align:center;"),
                          plotOutput("plt_g1", height = "360px")),
                   column(6, h5("Patron 2 en grilla", style = "text-align:center;"),
                          plotOutput("plt_g2", height = "360px"))
                 ),
                 br(),
                 h5("Superposicion en grilla", style = "text-align:center;"),
                 plotOutput("plt_gsuper", height = "360px")
        ),

        tabPanel("Perceptron Multicapa", br(),
                 fluidRow(
                   column(6,
                          div(style = "background:#e8f5e9;padding:12px;border-radius:8px;border:1px solid #4caf50;",
                              h4("Arquitectura", style = "margin-top:0;color:#1b5e20;"),
                              uiOutput("arq_red"))),
                   column(6,
                          div(style = "background:#e3f2fd;padding:12px;border-radius:8px;border:1px solid #1976d2;",
                              h4("Desempeno en Test (espacio condicional)",
                                 style = "margin-top:0;color:#0d47a1;"),
                              verbatimTextOutput("desempeno")))
                 ),
                 br(),
                 div(class = "metric-panel",
                     h4("Metricas binarias (matriz 2x2 — clase positiva: Coocurrencia)",
                        style = "margin-top:0;color:#0d47a1;text-align:center;"),
                     fluidRow(
                       column(4, uiOutput("card_accuracy")),
                       column(4, uiOutput("card_sensitivity")),
                       column(4, uiOutput("card_specificity"))
                     )
                 ),
                 br(),
                 fluidRow(
                   column(6,
                          div(style = "background:#fafafa;padding:12px;border-radius:8px;border:1px solid #bbb;",
                              h5("Matriz de Confusion 3x3 (espacio condicional)",
                                 style = "text-align:center;font-weight:bold;"),
                              tableOutput("mat_conf_3x3"))),
                   column(6,
                          div(style = "background:#fafafa;padding:12px;border-radius:8px;border:1px solid #bbb;",
                              h5("Matriz de Confusion 2x2 (Coocurrencia vs No Coocurrencia)",
                                 style = "text-align:center;font-weight:bold;"),
                              tableOutput("mat_conf_2x2")))
                 ),
                 br(),
                 fluidRow(
                   column(12,
                          div(style = "background:#fff3e0;padding:12px;border-radius:8px;border:1px solid #e65100;",
                              h4("Prediccion para el par actual (celdas coloreadas)",
                                 style = "margin-top:0;color:#bf360c;"),
                              fluidRow(
                                column(6, verbatimTextOutput("pred_actual")),
                                column(6, uiOutput("explica_diferencia"))
                              )))
                 ),
                 br(),
                 div(style = "background:#f3e5f5;padding:12px;border-radius:8px;border:1px solid #7b1fa2;",
                     h4("Variables de entrada por celda", style = "margin-top:0;color:#4a148c;"),
                     tableOutput("tbl_entradas"))
        ),

        tabPanel("Teoria", br(),
                 div(style = "max-width:900px;margin:auto;",
                     h3("La K de Ripley univariada"),
                     withMathJax(
                       "$$K(r) = \\lambda^{-1}\\,E[\\# \\text{ puntos en disco de radio }r]$$"),
                     p("Para CSR (Proceso de Poisson Homogeneo):"),
                     withMathJax("$$K_{teo}(r) = \\pi r^2$$"),
                     hr(),
                     h3("La K Cruzada"),
                     withMathJax(
                       "$$K_{AB}(r) = \\lambda_B^{-1}\\,E[\\# \\text{ puntos B en disco de radio }r \\text{ centrado en punto A}]$$"),
                     div(style = "background:#e8f5e9;padding:10px;border-radius:6px;margin-bottom:12px;",
                         tags$ul(
                           tags$li(strong("K12 > pi*r^2:"), " Atraccion: coocurrencia positiva"),
                           tags$li(strong("K12 = pi*r^2:"), " Independencia espacial"),
                           tags$li(strong("K12 < pi*r^2:"), " Repulsion: exclusion mutua")
                         )
                     ),
                     hr(),
                     h3("Evaluacion en espacio condicional"),
                     p("Las celdas blancas (sin P1 ni P2) se excluyen del test. La exactitud y las matrices de confusion se calculan solo sobre celdas con color."),
                     p("La matriz 2x2 colapsa Atraccion y Repulsion en Coocurrencia; las celdas con un solo patron se etiquetan como No Coocurrencia.")
                 )
        )
      )
    )
  )
)

# ============================================================
# SERVER
# ============================================================
server <- function(input, output, session) {

  rv <- reactiveValues(
    p1 = NULL, p2 = NULL, pp1 = NULL, pp2 = NULL,
    K1 = NULL, K2 = NULL, K12 = NULL,
    cls1 = NULL, cls2 = NULL, cls12 = NULL,
    k1v = NULL, k2v = NULL, k12v = NULL,
    modelo = NULL, k_max = NULL,
    acc = NULL, cm3 = NULL, cm2 = NULL,
    metricas = NULL, datos = NULL,
    n_test_coloreadas = NULL, n_test_total = NULL,
    pred_par = NULL
  )

  observe({
    max_coin <- min(input$n1, input$n2)
    updateSliderInput(
      session, "n_coincidentes",
      max   = max_coin,
      value = min(input$n_coincidentes, max_coin)
    )
  })

  observeEvent(input$simular, {
    withProgress(message = "Generando patrones y K de Ripley...", {
      rv$p1 <- generar_patron(input$tipo1, input$n1, input$seed1)
      rv$p2 <- generar_patron2_con_coincidencia(
        rv$p1, input$tipo2, input$n2,
        input$n_coincidentes, input$seed2
      )
      rv$pp1 <- hacer_ppp(rv$p1)
      rv$pp2 <- hacer_ppp(rv$p2)

      setProgress(0.4)

      rv$K1 <- tryCatch(Kest(rv$pp1, correction = "iso"), error = function(e) NULL)
      rv$K2 <- tryCatch(Kest(rv$pp2, correction = "iso"), error = function(e) NULL)
      ppc <- tryCatch(superimpose(A = rv$pp1, B = rv$pp2), error = function(e) NULL)
      rv$K12 <- tryCatch(
        Kcross(ppc, "A", "B", correction = "iso"),
        error = function(e) NULL
      )

      setProgress(0.8)

      if (!is.null(rv$K1))  rv$cls1  <- clasificar_univ(rv$K1)
      if (!is.null(rv$K2))  rv$cls2  <- clasificar_univ(rv$K2)
      if (!is.null(rv$K12)) rv$cls12 <- clasificar_biv(rv$K12)

      rv$k1v  <- if (!is.null(rv$K1))  extraer_K_val(rv$K1)  else NA
      rv$k2v  <- if (!is.null(rv$K2))  extraer_K_val(rv$K2)  else NA
      rv$k12v <- if (!is.null(rv$K12)) extraer_K_val(rv$K12) else NA
    })
  })

  observeEvent(input$entrenar, {
    withProgress(message = "Simulando datos de entrenamiento (nivel celda)...", {
      datos <- simular_datos_celdas(input$n_sim, 60)
      req(nrow(datos) > 0)

      k_max <- attr(datos, "k_max")
      if (is.null(k_max) || !is.finite(k_max)) {
        k_max <- max(c(datos$k1_n, datos$k2_n, datos$k12_n), na.rm = TRUE) + 1e-6
      }

      setProgress(0.5, "Particion train/test por simulacion...")

      sim_ids <- unique(datos$sim_id)
      set.seed(42)
      pct <- input$pct_train / 100
      n_train_sim <- max(1L, floor(pct * length(sim_ids)))
      train_sims <- sample(sim_ids, n_train_sim)
      train <- datos %>% filter(sim_id %in% train_sims)
      test  <- datos %>% filter(!sim_id %in% train_sims)

      train_col <- filtrar_celdas_coloreadas(train)
      test_col  <- filtrar_celdas_coloreadas(test)
      req(nrow(train_col) > 30, nrow(test_col) > 10)

      setProgress(0.75, "Entrenando perceptron multicapa...")

      modelo <- nnet(
        clase_f ~ k1_n + k2_n + k12_n + cls1_f + cls2_f + ocupa_p1 + ocupa_p2,
        data  = train_col,
        size  = input$size_nn,
        decay = as.numeric(input$decay_nn),
        maxit = 500,
        trace = FALSE
      )

      pred <- predict(modelo, test_col, type = "class")

      rv$cm3 <- table(
        Pred = pred,
        Real = as.character(test_col$clase_f),
        useNA = "no"
      )
      rv$metricas <- calcular_metricas_condicionadas(test_col$clase_f, pred)
      rv$cm2 <- rv$metricas$cm2
      rv$acc <- rv$metricas$accuracy

      rv$n_test_coloreadas <- nrow(test_col)
      rv$n_test_total      <- nrow(test)
      rv$modelo <- modelo
      rv$k_max  <- k_max
      rv$datos  <- datos
      rv$pred_par <- NULL
    })
  })

  observeEvent(input$predecir, {
    req(rv$modelo, rv$p1, rv$p2, rv$k1v, rv$k2v, rv$k12v,
        rv$cls1, rv$cls2, rv$cls12, rv$k_max)

    celdas <- construir_datos_celdas(
      rv$p1, rv$p2, rv$k1v, rv$k2v, rv$k12v,
      rv$cls1, rv$cls2, rv$cls12, rv$k_max
    )
    celdas_col <- filtrar_celdas_coloreadas(celdas)
    req(nrow(celdas_col) > 0)

    pred <- predict(rv$modelo, celdas_col, type = "class")
    prob <- predict(rv$modelo, celdas_col, type = "raw")

    rv$pred_par <- list(
      pred = pred,
      prob = prob,
      real = as.character(celdas_col$clase_f),
      n_ambos = sum(celdas_col$ocupa_p1 == 1L & celdas_col$ocupa_p2 == 1L),
      n_solo  = sum(xor(celdas_col$ocupa_p1 == 1L, celdas_col$ocupa_p2 == 1L)),
      clase_global_k = rv$cls12
    )
  })

  output$plt_p1 <- renderPlot({
    req(rv$p1)
    gg_puntos(rv$p1, "#1565C0", paste("Patron 1:", input$tipo1))
  })

  output$plt_p2 <- renderPlot({
    req(rv$p2)
    gg_puntos(rv$p2, "#B71C1C", paste("Patron 2:", input$tipo2))
  })

  output$plt_super <- renderPlot({
    req(rv$p1, rv$p2)
    d1 <- data.frame(gx = rv$p1$col, gy = rv$p1$row, pat = "Patron 1")
    d2 <- data.frame(gx = rv$p2$col, gy = rv$p2$row, pat = "Patron 2")
    df <- rbind(d1, d2)
    grilla_df <- expand.grid(gx = 1:30, gy = 1:30)
    ggplot() +
      geom_tile(data = grilla_df, aes(x = gx, y = gy),
                fill = "white", color = "gray85", linewidth = 0.25) +
      geom_point(data = df, aes(x = gx, y = gy, color = pat),
                 size = 3, alpha = 0.8) +
      scale_color_manual(values = c("Patron 1" = "#1565C0", "Patron 2" = "#B71C1C"),
                         name = NULL) +
      coord_equal() +
      scale_x_continuous(limits = c(0.5, 30.5)) +
      scale_y_continuous(limits = c(0.5, 30.5)) +
      labs(title = "Superposicion", x = NULL, y = NULL) +
      theme_minimal(base_size = 9) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5),
            legend.position = "bottom")
  })

  output$resumen_K <- renderText({
    req(rv$cls1, rv$cls2, rv$cls12)
    n_coinc <- sum(paste(rv$p1$col, rv$p1$row) %in% paste(rv$p2$col, rv$p2$row))
    K_teo <- round(pi * 0.04, 5)
    paste0(
      "Patron 1 : ", toupper(rv$cls1),
      "   |   K(r=0.2) = ", round(rv$k1v, 5),
      "\nPatron 2 : ", toupper(rv$cls2),
      "   |   K(r=0.2) = ", round(rv$k2v, 5),
      "\nCeldas coincidentes (P1 & P2) : ", n_coinc,
      " / slider = ", input$n_coincidentes,
      "\n", paste(rep("-", 52), collapse = ""),
      "\nCoocurrencia : ", toupper(rv$cls12),
      "   |   K12(r=0.2) = ", round(rv$k12v, 5),
      "\nK teorica CSR : ", K_teo
    )
  })

  output$plt_K1 <- renderPlot({ req(rv$K1); gg_K(rv$K1, "#1565C0", "K univariada — Patron 1") })
  output$plt_K2 <- renderPlot({ req(rv$K2); gg_K(rv$K2, "#B71C1C", "K univariada — Patron 2") })
  output$plt_K12 <- renderPlot({ req(rv$K12); gg_K(rv$K12, "#6A1B9A", "K cruzada — Coocurrencia") })

  output$interp_K <- renderUI({
    req(rv$cls12)
    col <- switch(rv$cls12,
                  atraccion = "#1b5e20",
                  repulsion = "#b71c1c",
                  sin_coocurrencia = "#0d47a1")
    msg <- switch(rv$cls12,
                  atraccion = "K12 > pi*r^2: ATRACCION — los patrones coocurren espacialmente",
                  repulsion = "K12 < pi*r^2: REPULSION — los patrones se evitan mutuamente",
                  sin_coocurrencia = "K12 ≈ pi*r^2: INDEPENDENCIA — sin asociacion espacial")
    tagList(
      tags$p(strong("Patron 1:"), rv$cls1, " | K =", round(rv$k1v, 5)),
      tags$p(strong("Patron 2:"), rv$cls2, " | K =", round(rv$k2v, 5)),
      tags$hr(),
      tags$p(strong("Coocurrencia:"),
             style = paste0("color:", col, ";font-size:1.1em;font-weight:bold;")),
      tags$p(msg, style = paste0("color:", col, ";"))
    )
  })

  output$plt_g1 <- renderPlot({
    req(rv$p1)
    gg_grilla(rv$p1, "#1565C0", paste("Grilla — Patron 1:", input$tipo1))
  })

  output$plt_g2 <- renderPlot({
    req(rv$p2)
    gg_grilla(rv$p2, "#B71C1C", paste("Grilla — Patron 2:", input$tipo2))
  })

  output$plt_gsuper <- renderPlot({
    req(rv$p1, rv$p2)
    g  <- expand.grid(gx = 1:30, gy = 1:30)
    e1 <- paste(g$gx, g$gy) %in% paste(rv$p1$col, rv$p1$row)
    e2 <- paste(g$gx, g$gy) %in% paste(rv$p2$col, rv$p2$row)
    g$estado <- ifelse(e1 & e2, "Ambos",
                       ifelse(e1, "Patron 1",
                              ifelse(e2, "Patron 2", "Vacio")))
    ggplot(g, aes(x = gx, y = gy, fill = estado)) +
      geom_tile(color = "gray80", linewidth = 0.25) +
      scale_fill_manual(values = c(
        "Patron 1" = "#1565C0",
        "Patron 2" = "#B71C1C",
        "Ambos"    = "#6A1B9A",
        "Vacio"    = "white")) +
      coord_equal() +
      labs(title = "Superposicion en Grilla 30x30",
           x = NULL, y = NULL, fill = NULL) +
      theme_minimal(base_size = 8) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5),
            legend.position = "bottom",
            panel.grid = element_blank())
  })

  output$arq_red <- renderUI({
    tagList(
      tags$p(strong("Entradas (7) por celda:")),
      tags$ul(
        tags$li("K1, K2, K12 en r=0.2 (simulacion)"),
        tags$li("Clase1, Clase2 univariada (simulacion)"),
        tags$li("ocupa_p1, ocupa_p2 (indicadores de celda)")
      ),
      tags$p(strong(paste0("Capa oculta: ", input$size_nn,
                           " neuronas (sigmoide)"))),
      tags$p(strong("Salida (3): atraccion / sin_coocurrencia / repulsion")),
      tags$p(paste0("Decay L2 (lambda): ", input$decay_nn)),
      tags$p(paste0("maxit: 500")),
      tags$p(paste0("Division: ", input$pct_train, "% train / ",
                    100 - input$pct_train, "% test (por simulacion)")),
      tags$p(em("Test evaluado solo en celdas coloreadas (espacio condicional)."))
    )
  })

  output$desempeno <- renderText({
    req(rv$acc, rv$datos, rv$n_test_coloreadas)
    n_sim <- length(unique(rv$datos$sim_id))
    paste0(
      "Exactitud condicionada : ", round(rv$acc * 100, 1), "%\n",
      "Celdas test (coloreadas): ", rv$n_test_coloreadas, "\n",
      "Celdas test (excluidas blancas): ",
      rv$n_test_total - rv$n_test_coloreadas, "\n",
      "Simulaciones totales   : ", n_sim, "\n",
      "size = ", input$size_nn,
      " | decay = ", input$decay_nn
    )
  })

  fmt_pct <- function(x) {
    if (is.na(x)) "N/A" else paste0(round(100 * x, 1), "%")
  }

  output$card_accuracy <- renderUI({
    req(rv$metricas)
    div(class = "metric-card",
        div(class = "metric-value", fmt_pct(rv$metricas$accuracy)),
        div(class = "metric-label", "Exactitud (Accuracy)"))
  })

  output$card_sensitivity <- renderUI({
    req(rv$metricas)
    div(class = "metric-card",
        div(class = "metric-value", fmt_pct(rv$metricas$sensitivity)),
        div(class = "metric-label", "Sensibilidad (Coocurrencia)"))
  })

  output$card_specificity <- renderUI({
    req(rv$metricas)
    div(class = "metric-card",
        div(class = "metric-value", fmt_pct(rv$metricas$specificity)),
        div(class = "metric-label", "Especificidad (No Coocurrencia)"))
  })

  render_cm_table <- function(cm) {
    df <- as.data.frame.matrix(cm)
    df$Pred <- rownames(df)
    df[, c("Pred", colnames(cm))]
  }

  output$mat_conf_3x3 <- renderTable({
    req(rv$cm3)
    render_cm_table(rv$cm3)
  }, bordered = TRUE, striped = TRUE, hover = TRUE)

  output$mat_conf_2x2 <- renderTable({
    req(rv$cm2)
    render_cm_table(rv$cm2)
  }, bordered = TRUE, striped = TRUE, hover = TRUE)

  output$pred_actual <- renderText({
    if (!is.null(rv$pred_par)) {
      pp <- rv$pred_par
      pred_moda <- names(sort(table(pp$pred), decreasing = TRUE))[1]
      paste0(
        "Clasificacion K Ripley (global): ", pp$clase_global_k, "\n",
        "Prediccion modal de la red   : ", pred_moda, "\n",
        "Celdas evaluadas (coloreadas): ", length(pp$pred), "\n",
        "  - Ambos patrones           : ", pp$n_ambos, "\n",
        "  - Un solo patron           : ", pp$n_solo, "\n\n",
        "Distribucion predicciones:\n",
        paste(names(table(pp$pred)), table(pp$pred),
              sep = " : ", collapse = "\n")
      )
    } else {
      req(rv$modelo, rv$p1, rv$p2, rv$k1v, rv$k2v, rv$k12v,
          rv$cls1, rv$cls2, rv$cls12, rv$k_max)
      celdas <- construir_datos_celdas(
        rv$p1, rv$p2, rv$k1v, rv$k2v, rv$k12v,
        rv$cls1, rv$cls2, rv$cls12, rv$k_max
      )
      celdas_col <- filtrar_celdas_coloreadas(celdas)
      pred <- predict(rv$modelo, celdas_col, type = "class")
      pred_moda <- names(sort(table(pred), decreasing = TRUE))[1]
      paste0(
        "Clasificacion K Ripley (global): ", rv$cls12, "\n",
        "Prediccion modal de la red   : ", pred_moda, "\n",
        "(Pulse 'Clasificar con la Red' para detalle por celda)"
      )
    }
  })

  output$explica_diferencia <- renderUI({
    req(rv$cls12)
    pred_lbl <- if (!is.null(rv$pred_par)) {
      names(sort(table(rv$pred_par$pred), decreasing = TRUE))[1]
    } else if (!is.null(rv$modelo)) {
      "pendiente"
    } else {
      return(tags$p("Entrene el modelo primero."))
    }
    if (pred_lbl == "pendiente") {
      return(tags$p("Pulse 'Clasificar con la Red' para comparar con K de Ripley."))
    }

    if (pred_lbl == rv$cls12) {
      div(style = "background:#e8f5e9;padding:10px;border-radius:6px;border:1px solid #4caf50;",
          tags$b(style = "color:#1b5e20;", "La red COINCIDE con K de Ripley (clase modal)"),
          tags$p(style = "color:#2e7d32;margin-top:6px;margin-bottom:0;",
                 "La prediccion predominante en celdas coloreadas coincide con la clasificacion bivariada global."))
    } else {
      div(style = "background:#fce4ec;padding:10px;border-radius:6px;border:1px solid #c62828;",
          tags$b(style = "color:#b71c1c;",
                 paste0("Diferencia modal: K=", toupper(rv$cls12),
                        " vs Red=", toupper(pred_lbl))),
          tags$p(style = "color:#555;margin-top:6px;margin-bottom:0;",
                 "La red clasifica celda a celda en el espacio condicional; la K de Ripley resume el par completo."))
    }
  })

  output$tbl_entradas <- renderTable({
    req(rv$k1v, rv$k2v, rv$k12v, rv$cls1, rv$cls2)
    data.frame(
      Variable = c("K1(r=0.2)", "K2(r=0.2)", "K12(r=0.2)",
                   "Clase1", "Clase2", "ocupa_p1", "ocupa_p2"),
      Valor = c(round(rv$k1v, 5), round(rv$k2v, 5), round(rv$k12v, 5),
                rv$cls1, rv$cls2, "0/1 por celda", "0/1 por celda"),
      Tipo = c("Cuantitativa (sim)", "Cuantitativa (sim)", "Cuantitativa (sim)",
               "Categorica (sim)", "Categorica (sim)",
               "Binaria (celda)", "Binaria (celda)")
    )
  }, bordered = TRUE, striped = TRUE)
}

shinyApp(ui = ui, server = server)

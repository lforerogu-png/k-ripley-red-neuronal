# Tablas generadas por el pipeline

### Distribución de clases (celdas coloreadas)

| clase            |   n_celdas |
|:-----------------|-----------:|
| repulsion        |      16160 |
| sin_coocurrencia |       6264 |
| atraccion        |       3231 |

### Comparación de esquemas de partición

| esquema   |   n_train |   n_test |   accuracy_test |
|:----------|----------:|---------:|----------------:|
| 70/15/15  |     17439 |     4132 |          0.9833 |
| 80/20     |     20158 |     5497 |          0.9874 |

### Validación cruzada 5-fold (media=0.994)

|   fold |   n_test |   accuracy |
|-------:|---------:|-----------:|
|      1 |     5136 |     1      |
|      2 |     5125 |     0.9858 |
|      3 |     5127 |     0.9975 |
|      4 |     5130 |     1      |
|      5 |     5137 |     0.9866 |

### Comparación de arquitecturas de MLP

| config              |   capas | arquitectura   |    lr |   loss_train_final |   loss_val_final |   acc_train_final |   acc_val_final |
|:--------------------|--------:|:---------------|------:|-------------------:|-----------------:|------------------:|----------------:|
| L1[8]_lr0.001       |       1 | (8,)           | 0.001 |             0.0001 |           0      |            1      |          1      |
| L1[8]_lr0.01        |       1 | (8,)           | 0.01  |             0      |           0      |            1      |          1      |
| L1[8]_lr0.1         |       1 | (8,)           | 0.1   |             0      |           0      |            1      |          1      |
| L1[16]_lr0.001      |       1 | (16,)          | 0.001 |             0      |           0      |            1      |          1      |
| L1[16]_lr0.01       |       1 | (16,)          | 0.01  |             0      |           0      |            1      |          1      |
| L1[16]_lr0.1        |       1 | (16,)          | 0.1   |             0.0008 |           0      |            0.9996 |          1      |
| L2[16x8]_lr0.001    |       2 | (16, 8)        | 0.001 |             0      |           0      |            1      |          1      |
| L2[16x8]_lr0.01     |       2 | (16, 8)        | 0.01  |             0      |           0      |            1      |          1      |
| L2[32x16]_lr0.001   |       2 | (32, 16)       | 0.001 |             0      |           0      |            1      |          1      |
| L2[32x16]_lr0.01    |       2 | (32, 16)       | 0.01  |             0      |           0      |            1      |          1      |
| L3[24x16x8]_lr0.01  |       3 | (24, 16, 8)    | 0.01  |             0      |           0      |            1      |          1      |
| L3[32x16x8]_lr0.001 |       3 | (32, 16, 8)    | 0.001 |             0      |           0      |            1      |          1      |
| L3[32x16x8]_lr0.01  |       3 | (32, 16, 8)    | 0.01  |             0      |           0      |            1      |          1      |
| L3[24x16x8]_lr0.001 |       3 | (24, 16, 8)    | 0.001 |             0      |           0      |            1      |          1      |
| L2[16x8]_lr0.1      |       2 | (16, 8)        | 0.1   |             0.2123 |           0.3161 |            0.9021 |          0.8144 |
| L2[32x16]_lr0.1     |       2 | (32, 16)       | 0.1   |             0.2187 |           0.3168 |            0.9009 |          0.8144 |
| L3[24x16x8]_lr0.1   |       3 | (24, 16, 8)    | 0.1   |             0.5767 |           0.6699 |            0.6949 |          0.6582 |
| L3[32x16x8]_lr0.1   |       3 | (32, 16, 8)    | 0.1   |             0.855  |           1.0067 |            0.6499 |          0.5889 |

### Entropía cruzada vs MSE

| perdida          |   accuracy_test |   loss_final |
|:-----------------|----------------:|-------------:|
| entropia_cruzada |          1      |       0      |
| mse              |          0.9833 |       0.0086 |

### Métricas por clase (3 clases, test condicional)

| clase            |   precision |   recall |    f1 |   support |
|:-----------------|------------:|---------:|------:|----------:|
| atraccion        |       1     |    0.911 | 0.954 |       778 |
| repulsion        |       1     |    1     | 1     |      2422 |
| sin_coocurrencia |       0.931 |    1     | 0.964 |       932 |

### Métricas modelo binario (test condicional)

|   accuracy |   sensibilidad |   especificidad |   precision |   f1 |
|-----------:|---------------:|----------------:|------------:|-----:|
|          1 |              1 |               1 |           1 |    1 |

### Regresión logística binaria (coeficientes)

| variable   |    coef |   std_err |   z |   p_valor |   ic_inf |   ic_sup |   odds_ratio | significativo   |
|:-----------|--------:|----------:|----:|----------:|---------:|---------:|-------------:|:----------------|
| const      | -2.4238 |       nan | nan |       nan |      nan |      nan |       0.0886 | False           |
| k1_n       | -0.3357 |       nan | nan |       nan |      nan |      nan |       0.7148 | False           |
| k2_n       | -0.1031 |       nan | nan |       nan |      nan |      nan |       0.902  | False           |
| k12_n      |  0.5268 |       nan | nan |       nan |      nan |      nan |       1.6935 | False           |
| cls1_f     | -0.1525 |       nan | nan |       nan |      nan |      nan |       0.8586 | False           |
| cls2_f     | -0.0254 |       nan | nan |       nan |      nan |      nan |       0.9749 | False           |
| ocupa_p1   |  6.9077 |       nan | nan |       nan |      nan |      nan |     999.97   | False           |
| ocupa_p2   |  6.3445 |       nan | nan |       nan |      nan |      nan |     569.349  | False           |

### Selección de variables por Lasso (L1)

| variable   |   coef |   abs_coef | seleccionada   |
|:-----------|-------:|-----------:|:---------------|
| ocupa_p1   | 1.7149 |     1.7149 | True           |
| ocupa_p2   | 1.5228 |     1.5228 | True           |
| k1_n       | 0      |     0      | False          |
| k12_n      | 0      |     0      | False          |
| k2_n       | 0      |     0      | False          |
| cls2_f     | 0      |     0      | False          |
| cls1_f     | 0      |     0      | False          |

### Importancia por permutación (MLP)

| variable   |   importancia_media |   importancia_std |
|:-----------|--------------------:|------------------:|
| ocupa_p1   |              0.3345 |            0.0049 |
| ocupa_p2   |              0.3082 |            0.0051 |
| k12_n      |              0.1154 |            0.0045 |
| cls1_f     |              0.0323 |            0.0019 |
| cls2_f     |              0.0203 |            0.002  |
| k1_n       |              0.0009 |            0.0011 |
| k2_n       |              0      |            0      |

### Ranking permutación (MLP) vs |coef| logístico

| variable   |   importancia_media |   rank_mlp |   abs_coef |   rank_logit |
|:-----------|--------------------:|-----------:|-----------:|-------------:|
| cls1_f     |              0.0323 |          4 |     0.1525 |            6 |
| cls2_f     |              0.0203 |          5 |     0.0254 |            8 |
| const      |            nan      |        nan |     2.4238 |            3 |
| k12_n      |              0.1154 |          3 |     0.5268 |            4 |
| k1_n       |              0.0009 |          6 |     0.3357 |            5 |
| k2_n       |              0      |          7 |     0.1031 |            7 |
| ocupa_p1   |              0.3345 |          1 |     6.9077 |            1 |
| ocupa_p2   |              0.3082 |          2 |     6.3445 |            2 |

### Índice de Moran global: agrupado vs aleatorio

| patron    |   moran_I |   p_valor | interpretacion                                         |
|:----------|----------:|----------:|:-------------------------------------------------------|
| agrupado  |    0.1905 |     0.002 | autocorrelación positiva: valores similares se agrupan |
| aleatorio |   -0.0204 |     0.238 | sin dependencia espacial significativa (p >= 0.05)     |

### MLP vs Random Forest vs SVM

| modelo       |   accuracy |   f1_macro |   f1_weighted |
|:-------------|-----------:|-----------:|--------------:|
| MLP(16x8)    |     0.9833 |     0.9726 |        0.9832 |
| SVM(rbf)     |     0.9833 |     0.9726 |        0.9832 |
| RandomForest |     0.9789 |     0.9654 |        0.9788 |

### Robustez ante ruido en las K

|   nivel_ruido |   accuracy |
|--------------:|-----------:|
|          0    |     0.9833 |
|          0.05 |     0.9833 |
|          0.1  |     0.9828 |
|          0.2  |     0.9806 |
|          0.3  |     0.9748 |
|          0.5  |     0.9724 |

### Efecto de early stopping y L2 (alpha)

|   alpha | early_stopping   |   acc_train |   acc_test |    gap |
|--------:|:-----------------|------------:|-----------:|-------:|
|   0     | False            |      1      |     0.9833 | 0.0167 |
|   0     | True             |      1      |     0.9833 | 0.0167 |
|   0.001 | False            |      1      |     0.9833 | 0.0167 |
|   0.001 | True             |      1      |     0.9833 | 0.0167 |
|   0.1   | False            |      1      |     0.9833 | 0.0167 |
|   0.1   | True             |      1      |     0.9833 | 0.0167 |
|   1     | False            |      0.9984 |     0.9833 | 0.0151 |
|   1     | True             |      0.9982 |     0.9743 | 0.0238 |

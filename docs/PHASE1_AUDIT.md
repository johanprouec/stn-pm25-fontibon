---
name: PHASE1_AUDIT
description: Audit of the Primera Entrega notebook/report and of both raw datasets, produced before starting Phase 2 work
---

# Auditoría de la Primera Entrega

Este documento registra lo que se encontró al revisar `PrimeraEntrega_DL.ipynb` (código y
salidas), `Primer Entrega DL.pdf` y los dos datasets locales, antes de escribir cualquier
código nuevo para la Segunda Entrega. Todos los números aquí fueron re-ejecutados de forma
independiente con `scripts/run_baseline_fontibon.py` y los scripts de auditoría de datasets;
no se transcribieron a ciegas del PDF.

## 1. Decisiones reproducidas correctamente

Verificado ejecutando el pipeline refactorizado (`src/data/fontibon.py`,
`src/evaluation/metrics.py`) contra el CSV original y comparando contra el notebook y el PDF:

| Elemento | Valor encontrado | Fuente | Estado |
|---|---|---|---|
| Estación | FONTIBON (17 520 filas; existe también `MOVIL FONTIBON`, no usada) | notebook cell 2 | ✅ coincide |
| Variables armonizadas | pm25, temp, dew (punto de rocío calculado) | notebook cell 2 | ✅ coincide |
| Fórmula punto de rocío | Magnus, a=17.625, b=243.04 | notebook cell 2 | ✅ coincide |
| Ventana | 24 horas | notebook cell 2 | ✅ coincide |
| Partición | 70/15/15 por posición de fila (no por fecha fija) | notebook cell 4 | ✅ coincide |
| Ventanas train/val/test | 10 802 / 2 564 / 2 574 | notebook cell 4 output, Tabla 2 del PDF | ✅ coincide exactamente |
| Fechas de partición | train 2021-01-01→2022-05-26 23:00, val 2022-05-27→2022-09-13 11:00, test 2022-09-13 12:00→2022-12-31 23:00 | notebook cell 4 output | ✅ coincide |
| PM2.5 objetivo nunca se imputa | confirmado en `construir_ventanas`: `if pd.isna(parte.iloc[k]["pm25"]): continue` | notebook cell 4 | ✅ coincide |
| Interpolación de entradas | solo bloques ≤ 6 horas, con valores observados a ambos lados | notebook cell 4 | ✅ coincide |
| Chequeo causal anti-fuga | ninguna interpolación usada en el historial puede tener `extremo_derecho >= k` (el objetivo) | notebook cell 4 | ✅ coincide, verificado también con test sintético (`tests/test_windows_no_cross_partition.py`) |
| Ridge — definición exacta | `make_pipeline(StandardScaler(), Ridge(alpha=1.0))`, `StandardScaler` ajustado únicamente con `Xtr` | notebook cell 6 | ✅ coincide |
| Features que recibe Ridge | 72 = 24 horas × 3 variables, aplanado (`aplanar`/`flatten`) | notebook cell 6, texto del PDF §3 | ✅ coincide |
| Selección de línea base | únicamente con RMSE de validación, test reservado | notebook cell 6-8 | ✅ coincide |
| Resultado validación Ridge | RMSE=6.545, MAE=4.783, R²=0.587 | notebook cell 6 output, Tabla 3 PDF | ✅ coincide (6.5448..., 4.7829..., 0.58675...) |
| Resultado test Ridge | RMSE=7.063, MAE=5.115, R²=0.593 | notebook cell 8 output, PDF §6.4 | ✅ coincide (7.0632..., 5.1146..., 0.59350...) |
| Semilla | 42 (`SEED = 42`, `np.random.seed(SEED)`, `random_state=SEED` en Random Forest) | notebook cell 2 | ✅ única semilla usada en Fase 1; no hay convención previa de 3 semillas |

**Conclusión:** el refactor en `src/` reproduce el pipeline de la Primera Entrega de forma
exacta (`scripts/run_baseline_fontibon.py` incluye aserciones automáticas que comparan contra
estos números y actualmente pasan).

## 2. Inconsistencias encontradas

### 2.1 Umbral de episodios altos: 40 (informe) vs 37 (notebook ejecutado) — **relevante**

- El **informe PDF** (`Primer Entrega DL.pdf`, p.4) dice textualmente: *"se utiliza un umbral
  de 40 µg/m³"* y reporta: 118 observaciones, PM2.5 real medio 49.87, predicho medio 34.91,
  subestimación media 14.96 µg/m³.
- El **notebook ejecutado** (`PrimeraEntrega_DL.ipynb`, celda 10) usa literalmente
  `umbral = 37`, con el comentario *"norma diaria colombiana de PM2.5 según la Resolución 2254
  de 2017"*, y su salida real es: 159 observaciones, real medio 46.88, predicho medio 33.88,
  subestimación media 13.00 µg/m³.
- **Verificación independiente** (`verify_notebook.py`, re-ejecutado sobre el CSV original):
  con `umbral=40` se obtienen exactamente 118 / 49.87 / 34.91 / 14.96 — es decir, los números
  del informe corresponden exactamente a 40, no a 37. Con `umbral=37` se reproduce exactamente
  la salida de la celda del notebook (159 / 46.88 / 33.88 / 13.00).
- **Interpretación:** el notebook fue editado después de escribirse el informe, cambiando el
  umbral exploratorio de 40 a 37 sin actualizar el texto del informe. Ambos documentos son
  internamente consistentes consigo mismos, pero no entre sí.
- **Decisión tomada (por instrucción explícita del usuario):** se mantiene 40 µg/m³ como
  umbral principal en todo el código nuevo (`src/evaluation/metrics.py:DEFAULT_HIGH_THRESHOLD`),
  por continuidad metodológica con el informe oficial. `scripts/run_baseline_fontibon.py`
  calcula y guarda ambos umbrales (40 y 37) explícitamente en
  `outputs/metrics/baseline_fontibon_ridge.json`, sin sustituir uno por otro silenciosamente.
- **No corregido:** no se modificó el notebook ni el informe originales; esta sección
  documenta la discrepancia para que el equipo decida si el informe de Primera Entrega
  necesita una fe de erratas.

### 2.2 Nombre de estación "Móvil Fontibón" vs "MOVIL FONTIBON"

El informe usa el nombre con tilde y mayúscula inicial ("Móvil Fontibón"); el CSV usa
`MOVIL FONTIBON` (sin tilde, todo mayúsculas). Es una diferencia puramente tipográfica en el
reporte, sin efecto en el código (la estación `MOVIL FONTIBON` no se usa de todas formas).

### 2.3 Redondeo en el texto del informe

El texto del PDF redondea a 3 decimales (p.ej. RMSE=6,545); los valores exactos calculados son
6.544817, 4.782950, 0.586752 (validación) y 7.063201, 5.114589, 0.593495 (test). No es una
inconsistencia, solo una precisión a anotar para quien compare cifras exactas en tests.

## 3. Riesgos de fuga de información (leakage) — evaluados

| Riesgo | Evaluación | Evidencia |
|---|---|---|
| Partición hecha después de imputar/escalar | **No ocurre.** El corte 70/15/15 se aplica sobre `X` cruda (antes de cualquier interpolación), y `construir_ventanas` interpola *dentro* de cada partición ya cortada. | `tests/test_partitions.py`, `tests/test_windows_no_cross_partition.py::test_real_partitions_windows_stay_within_bounds` |
| Scaler ajustado con datos fuera de train | **No ocurre.** `StandardScaler` se ajusta solo con `Xtr` dentro del `Pipeline`. | `tests/test_scaler_leakage.py` |
| Ventanas que cruzan fronteras train/val/test | **No puede ocurrir por construcción**, porque `construir_ventanas` opera sobre `X_original.iloc[inicio:fin]`, un slice ya acotado a una partición. | `tests/test_windows_no_cross_partition.py`, `tests/test_partitions.py::test_no_timestamp_overlap_between_partitions` |
| Interpolación de una hora usando el propio objetivo o el futuro | **Mitigado explícitamente**: el código registra `extremo_derecho` (índice del ancla derecha usada en cada interpolación) y descarta cualquier ventana cuyo historial haya usado un ancla `>= k` (la hora objetivo). Se agregó un test sintético que fuerza este caso límite. | `tests/test_windows_no_cross_partition.py::test_synthetic_window_rejects_interpolation_touching_target_hour` |
| PM2.5 objetivo imputado | **No ocurre.** El objetivo siempre sale de `parte.iloc[k]["pm25"]` (dato crudo); si es NaN, la ventana se descarta. | `tests/test_target_leakage.py` |
| Selección de línea base usando test | **No ocurre** en el notebook: `mejor = tabla_val["RMSE"].idxmin()` se calcula antes de tocar `Xte/yte`. | inspección de celdas 6–8 |

**Ningún riesgo de fuga fue encontrado sin mitigar.** Los 20 tests en `tests/` pasan.

## 4. Parámetros encontrados en el notebook

```
SEED = 42
VENTANA = 24
HORIZONTE = 1
LIMITE_IMPUTA = 6          # horas máximas de gap interpolable en variables de entrada
split = 70% / 15% / 15%    # por posición de fila, cronológico
Ridge(alpha=1.0)           # sin búsqueda de hiperparámetros; alpha fijo
RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
umbral episodios altos = 37 (celda ejecutada) vs 40 (texto del informe) — ver §2.1
```

No hay búsqueda de hiperparámetros (`GridSearch`/`CV`) para Ridge ni Random Forest en la
Primera Entrega: los valores (`alpha=1.0`, `n_estimators=300`, `min_samples_leaf=2`) están
fijados directamente en el código, sin justificación explícita de por qué esos valores y no
otros. Esto no es un error, pero es una limitación a mencionar si se pregunta por la
metodología de selección de la línea base.

## 5. Lógica de preprocesamiento (resumen ejecutable)

1. Filtrar el CSV a `STATION == "FONTIBON"`.
2. Construir timestamp horario (`DATE` + `HOUR`), ordenar, indexar por tiempo.
3. Parsear PM2.5, Temperature y RH a numérico (reemplazando `,` por `.`, coerción a NaN si
   falla).
4. Calcular punto de rocío vía Magnus a partir de `temp` y `hr`.
5. Cortar cronológicamente 70/15/15 por posición de fila (sobre las 17 520 horas totales).
6. Dentro de cada partición: interpolar linealmente solo huecos internos ≤ 6 horas con datos
   observados a ambos lados; registrar el ancla derecha usada.
7. Construir ventanas de 24 horas: descartar si el objetivo (`pm25` en `k`) es NaN, si el
   historial interpolado sigue teniendo NaN, o si alguna interpolación del historial usó una
   ancla `>= k`.
8. Aplanar cada ventana (24×3 → 72) para los modelos clásicos.
9. Ajustar `StandardScaler` + `Ridge(alpha=1.0)` solo con train; evaluar en validación primero,
   fijar el modelo ganador, evaluar en test al final.

Refactorizado sin cambios de lógica en `src/data/fontibon.py` y `src/evaluation/metrics.py`.

## 6. Auditoría del dataset Beijing (`PRSA_data_2010.1.1-2014.12.31.csv`)

| Característica | Valor |
|---|---|
| Filas | 43 824 |
| Periodo | 2010-01-01 00:00 → 2014-12-31 23:00, horario, sin huecos ni duplicados en el índice temporal |
| Columnas | No, year, month, day, hour, pm2.5, DEWP, TEMP, PRES, cbwd, Iws, Is, Ir (13 columnas) |
| PM2.5 faltante | 2 067 (4.72 %) |
| Bloques de faltantes en PM2.5 | 214 bloques; mediana 1h, máximo 155h; 20 bloques > 24h, 56 bloques > 6h |
| `cbwd` (dirección del viento) | Categórica: SE (15 290), NW (14 150), cv (9 387), NE (4 997) — **4 categorías, no ordinal** |
| PM2.5 | min 0, max 994, media 98.6 |
| DEWP | min −40, max 28 |
| TEMP | min −19, max 42 |
| PRES | min 991, max 1046 |
| Iws (viento acumulado) | min 0.45, max 585.6 |
| Is / Ir (nieve/lluvia acumulada) | mayormente 0; valores >0 en 368 / 1808 filas respectivamente |

**Comparación con el paper:** el paper reporta ~43 800 registros horarios 2010–2014; el
archivo real tiene 43 824, consistente. El paper no detalla el número exacto de faltantes de
PM2.5 en el cuerpo principal; el informe de Primera Entrega sí reporta 2 067 (4.72 %), que
coincide exactamente con esta auditoría independiente.

**Implicación de diseño:** `cbwd` debe codificarse con one-hot (`src/data/beijing.py:
one_hot_wind_direction`), nunca como entero ordinal — ver
`docs/IMPLEMENTATION_ASSUMPTIONS.md`.

## 7. Auditoría del dataset Fontibón (`Raw_Consolidated_ECA_Complete_2022.csv`)

| Característica | Valor |
|---|---|
| Filas totales | 35 040 (2 estaciones × 17 520 c/u) |
| Estaciones | `FONTIBON` (17 520), `MOVIL FONTIBON` (17 520) |
| Periodo (FONTIBON) | 2021-01-01 00:00 → 2022-12-31 23:00, horario, sin huecos ni duplicados |
| PM2.5 faltante | 923 (5.27 %), en 184 bloques, mediana 1h, máximo 111h, 6 bloques > 24h, 30 bloques > 6h |
| Temperature faltante | 11 (0.06 %) |
| RH faltante | 11 (0.06 %) |
| Baro_Pressure faltante | 4 942 (28.21 %) — confirma la exclusión de presión del informe |
| Wind_Speed faltante | 1 711 (9.77 %) |
| Wind_Direction | continua en grados (0–360), **no categórica** — confirma que no es comparable con `cbwd` de Beijing |
| PM2.5 media anual 2021 | 17.29 µg/m³ — coincide exactamente con el informe |
| PM2.5 media anual 2022 | 19.31 µg/m³ — coincide exactamente con el informe |
| PM2.5 rango | 0.1 – 100.8 µg/m³ |

**Aclaración explícita**: todo el proyecto (línea base Ridge, STN, análisis de distancia de
dominio) usa exclusivamente `STATION == "FONTIBON"` (estación fija). `MOVIL FONTIBON` nunca se
incluye, ni se combinan ambas estaciones en ningún momento — fijado en código
(`src/data/fontibon.py`, constante `STATION = "FONTIBON"`), no es una elección manual por
corrida.

No se encontraron valores fuera de rango físico plausible en PM2.5, temperatura o humedad
relativa para Fontibón. La columna `OZONE_ppb` (no usada en este proyecto) muestra valores
enteros grandes que sugieren un problema de separador decimal en la fuente original (p.ej.
`8943` en vez de `89.43`); como esta variable no forma parte del conjunto armonizado
(pm25/temp/dew), no se investigó ni se corrigió, pero se deja anotado por si se usa en el
futuro.

## 8. Ambigüedades del paper (Zhang & Zhang, 2023)

Ver `docs/IMPLEMENTATION_ASSUMPTIONS.md` para la lista completa con la separación explícita
entre reportado / inferido / asumido. Resumen: el paper especifica arquitectura general
(encoder-decoder, 3 capas encoder, 2 decoder, kernel Conv1D=3, pooling stride=2), optimizador,
tasa de aprendizaje, batch size, épocas máximas y loss, pero **no especifica** d_model, número
de cabezas de atención, d_ff, dropout, el factor exacto de selección Top-u de la atención
dispersa, longitud de entrada del decoder, tokens de inicio, criterio exacto de outliers,
inicialización de pesos, semilla, patience ni min_delta.

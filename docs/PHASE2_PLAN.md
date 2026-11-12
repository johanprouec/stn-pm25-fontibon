---
name: PHASE2_PLAN
description: Traceability matrix from the course guide's Segunda Entrega requirements to experiments, code and evidence, plus the approved Beijing validation split and the domain-distance plan — nothing here has been executed yet except what's marked done
---

# Plan de la Segunda Entrega (Fase 2)

## Estado

- **Architecture validated**: `SparseAttention` y `STNModel` (CANDIDATE_V1) implementados y
  probados (arquitectura + smoke test con datos sintéticos). Semillas únicas por bloque
  (`derive_child_seed`).
- **Beijing preprocessing validated**: `prepare_beijing_r1` implementado y probado (24 tests,
  `tests/test_beijing_pipeline.py`); auditoría en
  `outputs/metrics/beijing_preprocessing_audit.json`. Split cronológico train/validation/test
  aprobado, escalado y codificación one-hot ajustados solo con train, protocolo de faltantes
  causal-seguro.
- **R1 pilot completed**: `scripts/run_r1_pilot.py`, ≤5 épocas sobre un subconjunto cronológico
  de train/validation (rotulado explícitamente "PILOT DE INTEGRACIÓN — NO RESULTADO FINAL").
  Verificó la cadena completa de extremo a extremo; no midió desempeño final.
- **R1 development train/validation COMPLETED — las tres semillas (42, 123, 2026)**:
  `scripts/run_r1_development.py <seed>`, entrenamiento completo (sin subsampling: 29 237
  ventanas train, 2 841 validation) con `configs/beijing_r1_development.yaml`, idéntico entre
  semillas, cada una con inicialización independiente (nunca se reutilizan pesos de otra
  semilla). Trainer endurecido: métricas de entrenamiento deterministas separadas de la
  pérdida de optimización, evaluación por batches, semilla fijada antes de construir el
  modelo, historial guardado incrementalmente cada época, predicciones de validación
  fila-por-fila guardadas por semilla.

  | Seed | Best epoch | Stopping epoch | Val RMSE (µg/m³) | Val MAE | Val R² |
  |---|---|---|---|---|---|
  | 42 | 43 | 58 | 21.122 | 12.675 | 0.9474 |
  | 123 | 72 | 87 | 20.781 | 12.431 | 0.9491 |
  | 2026 | 52 | 67 | 21.242 | 12.893 | 0.9468 |
  | **media ± std** | — | — | **21.048 ± 0.240** | **12.666 ± 0.231** | **0.9477 ± 0.0012** |

  Persistence (referencia, no depende de semilla): RMSE=22.858, MAE=13.213, R²=0.9384. Las
  tres semillas superan a persistence de forma consistente (clasificación descriptiva "A" en
  las tres) con muy baja dispersión entre ellas (CV del RMSE ≈1.1%) — buena señal de
  estabilidad de la arquitectura CANDIDATE_V1. Resultados completos en
  `outputs/metrics/R1_development_seed{42,123,2026}_summary.json` y
  `outputs/metrics/R1_development_seed_comparison.json`. **No es tuning ni el R1 final** — ver
  "Criterio de sanity-check" más abajo. La comparación contra el paper (RMSE=19.04) sigue
  pendiente, deliberadamente, hasta nueva autorización.
- **R1 architecture = FROZEN**: arquitectura, hiperparámetros, preprocesamiento, particiones y
  los tres checkpoints (`R1_development_seed{42,123,2026}`) quedaron congelados y documentados
  en `outputs/metrics/R1_frozen_protocol.json` **antes** de tocar Beijing test 2014.
- **R1 validation = COMPLETE** (ver tabla de las tres semillas arriba).
- **R1 Beijing test 2014 = FINAL EVALUATION COMPLETE** — evaluado **una sola vez**, por
  inferencia pura (`training=False`, sin optimizador, sin gradientes) sobre los tres
  checkpoints ya congelados. `scripts/run_r1_test_evaluation.py`. 8 487 ventanas de test
  (coincide exactamente con lo auditado).

  | Seed | Test RMSE (µg/m³) | Test MAE | Test R² | Bias residual |
  |---|---|---|---|---|
  | 42 | 20.692 | 11.744 | 0.9508 | −1.476 |
  | 123 | 21.207 | 11.780 | 0.9483 | +0.462 |
  | 2026 | 21.481 | 12.060 | 0.9469 | −3.111 |
  | **media ± std** | **21.127 ± 0.400** | **11.862 ± 0.173** | **0.9487 ± 0.0019** | — |

  Persistence (test, sanity check): RMSE=22.024, MAE=11.915, R²=0.9442. Referencia publicada
  (Zhang & Zhang 2023, Beijing t+1): RMSE=19.04, MAE=11.13, R²=0.937 — **comparación
  descriptiva, no réplica exacta** (ver diferencias metodológicas documentadas en
  `docs/IMPLEMENTATION_ASSUMPTIONS.md`). Nuestro RMSE medio queda ~2.09 µg/m³ por encima del
  valor publicado; nuestro R² medio (0.9487) queda ligeramente por encima del publicado
  (0.937). Resultados completos en `outputs/metrics/R1_test_final_summary.json` y
  `R1_test_seed{42,123,2026}_summary.json`.

  **R1 no se modificará después de este punto.** No hubo entrenamiento durante esta
  evaluación, ninguna semilla se seleccionó mirando test, y la arquitectura/hiperparámetros no
  cambiaron después de ver los resultados.
- **R2 preprocessing = COMPLETE**: `prepare_beijing_r2` (`src/data/beijing.py`), 3 variables
  compartidas (`pm25`, `temp`, `dew`), orden fijado por
  `src/data/transfer_features.py::TRANSFER_FEATURE_NAMES` y **verificado contra el pipeline real
  de Fontibón** (`tests/test_beijing_r2_pipeline.py::test_beijing_r2_order_matches_fontibon_order`).
  Mismo split cronológico train/validation de R1 (Beijing test 2014 no se construye por
  defecto). Escaladores propios (`BeijingR2Scalers`), ajustados solo con train Beijing,
  independientes de los de Fontibón. 18 tests nuevos, todos verdes.
- **R2 compatibilidad estructural = AUDITADA, `compatible: true`**:
  `src/models/transfer_compat.py::compare_transfer_weight_shapes` comparó un modelo fuente
  Beijing-R2 (3 canales) contra un modelo con forma Fontibón (3 canales), ambos sin entrenar —
  96/96 pesos de encoder y 27/27 de decoder coinciden en forma
  (`outputs/metrics/R2_preprocessing_audit.json`). `output_head` es informativo, se reinicializa
  siempre para el destino.
- **R2 pilot de integración = EJECUTADO UNA VEZ, no es resultado final**:
  `scripts/run_r2_pilot.py`, semilla=42, subconjunto cronológico de 4096 ventanas train / 1024
  validación, 5 épocas, modelo CANDIDATE_V1 entrenado **desde cero** (no reutiliza checkpoints de
  R1 — ver `docs/IMPLEMENTATION_ASSUMPTIONS.md` §"R2", incompatibilidad de forma en la
  proyección de entrada 11 canales vs 3). Rotulado explícitamente "R2 PILOT DE INTEGRACION -- NO
  RESULTADO FINAL" en `outputs/metrics/R2_pilot_summary.json`. Verificó que la cadena completa
  funciona de punta a punta (loss y RMSE escalados bajando de forma monótona, norma de gradiente
  estabilizándose). Beijing test 2014 y Fontibón (validación/test) no se usaron.
- **R2 full source training = COMPLETE — las tres semillas (42, 123, 2026)**:
  `scripts/run_r2_source_training.py`, entrenamiento completo sin subsampling (29 237 ventanas
  train, 2 841 validation, idénticas a R1/R2 preprocessing), arquitectura y protocolo
  congelados, idéntico entre semillas, cada una con inicialización independiente (`set_experiment_seed(seed)`
  antes de construir el modelo — nunca se reutilizan pesos de otra semilla ni de R1). El proceso
  se interrumpió una vez por un reinicio del equipo justo al terminar seed 42 y arrancar
  seed 123 (ver `docs/IMPLEMENTATION_ASSUMPTIONS.md` §"R2 full source training", auditoría
  pasiva + relanzamiento); seed 42 nunca se reentrenó — su checkpoint/summary/metadata originales
  se conservaron intactos y solo se relanzaron seed 123 y seed 2026 desde cero.

  | Seed | Best epoch | Stopping epoch | Val RMSE (µg/m³) | Val MAE | Val R² | Bias | Compatible con target Fontibón |
  |---|---|---|---|---|---|---|---|
  | 42 | 54 | 69 | 21.141 | 13.007 | 0.9473 | +1.634 | true |
  | 123 | 58 | 73 | 20.971 | 12.544 | 0.9481 | +1.329 | true |
  | 2026 | 58 | 73 | 21.068 | 12.646 | 0.9476 | −0.207 | true |
  | **media ± std** | — | — | **21.060 ± 0.085** | **12.732 ± 0.243** | **0.9477 ± 0.0004** | — | — |

  Dispersión entre semillas muy baja (CV del RMSE ≈0.4%, incluso menor que R1's ≈1.1%) — buena
  señal de estabilidad de CANDIDATE_V1 también con el input reducido de 3 canales. Las tres
  semillas se conservan como checkpoints SOURCE (`outputs/checkpoints/R2_source_seed{42,123,2026}/best.weights.h5`),
  **ninguna se selecciona como "ganadora"** — la transferencia futura a Fontibón (Config B/C/D,
  no implementada) emparejará cada semilla local con su checkpoint source del mismo número de
  semilla (local seed 42 ← R2 source seed 42, etc.), nunca eligiendo el checkpoint source con
  menor RMSE de validación para inicializar todas las corridas locales — ver
  `docs/IMPLEMENTATION_ASSUMPTIONS.md`. Compatibilidad encoder/decoder con un modelo target con
  forma de Fontibón (3 canales) reverificada **desde el checkpoint guardado en disco** (no desde
  el modelo en memoria) para las tres semillas: `compatible: true` en los tres casos
  (`outputs/metrics/R2_source_seed{42,123,2026}_summary.json`,
  `outputs/metrics/R2_source_seed_comparison.json`). R2 no es una nueva réplica de R1 y no se
  compara contra el RMSE publicado del paper (19.04) como objetivo experimental. Beijing test
  2014 no usado en ninguna semilla; Fontibón no entrenado ni evaluado en ninguna semilla.
- **Fontibón STN pipeline = COMPLETE, NO entrenado/evaluado**:
  `src/data/fontibon.py::prepare_fontibon_stn`, mismas 3 variables/orden que R2
  (`TRANSFER_FEATURE_NAMES`), mismo split 70/15/15 ya aprobado para Ridge
  (`chronological_cuts` — train=10802, validation=2564, test=2574, reproducidos exactamente),
  escalador MinMax **propio de Fontibón** (`FontibonSTNScalers`, ajustado solo con train,
  nunca el de Beijing — ver `docs/IMPLEMENTATION_ASSUMPTIONS.md`). Test de Fontibón solo se
  construye dentro de un test dedicado que verifica el conteo (10802/2564/2574); nunca se
  construye en `src/training/transfer.py` ni se evalúa en ningún lado.
- **Infraestructura de transferencia A/B/C/D (`src/training/transfer.py`) = COMPLETE,
  NO EJECUTADA contra datos reales**: `TransferSTNModel` (congela encoder/decoder por
  submódulo y fuerza modo de inferencia — `training=False`, Dropout y muestreo de
  SparseAttention deterministas — en las partes congeladas, independientemente del `training`
  que reciba el modelo completo), `load_r2_source_into_target` (copia únicamente
  `.encoder`/`.decoder` desde un checkpoint R2 verificando antes compatibilidad vía
  `compare_transfer_weight_shapes`; `output_head` nunca se toca), `DifferentialLRTrainer`
  (únicamente para Config D — dos optimizadores Adam, uno por grupo de variables, sin modificar
  `Trainer` base; A/B/C siguen usando `Trainer` sin cambios). 22 tests nuevos, todos en
  arquitecturas sintéticas pequeñas o datos sintéticos — ningún test entrena sobre Beijing o
  Fontibón reales.
- **Auditoría del estado real (checkpoints, rutas, configs) = COMPLETE**: los tres checkpoints
  R2 se cargaron efectivamente desde disco en un modelo fresco (no solo se verificó que el
  archivo existiera) — 125/125 variables cambiaron respecto a la inicialización aleatoria,
  99 425 parámetros, coincide con cada metadata. Se encontraron y corrigieron 2 inconsistencias
  reales (ver `docs/IMPLEMENTATION_ASSUMPTIONS.md` y los propios `configs/*.yaml` para el detalle
  completo): (1) `configs/beijing_transfer_source_r2.yaml` seguía con `status: pilot_not_final`
  y `full_training.status: NOT_STARTED` después de que el entrenamiento completo ya había
  terminado; (2) `configs/transfer_{B,C,D}_*.yaml` apuntaban los tres a
  `outputs/checkpoints/r2_beijing_transfer_source/` (minúsculas, sin `{seed}`, una sola ruta
  compartida) — una ruta que nunca existió y que además habría roto el emparejamiento por
  semilla; corregido a `outputs/checkpoints/R2_source_seed{seed}/best.weights.h5` en los tres.
- **Análisis de distancia de dominio (sección 4) = EJECUTADO**: `scripts/run_domain_distance_analysis.py`,
  solo train de ambos dominios, nunca Beijing test 2014 ni Fontibón validation/test. Resultados en
  `outputs/metrics/domain_distance_analysis.json` y `outputs/figures/domain_distance/`: PM2.5
  medio Beijing train = 99.04 µg/m³ (std 91.66, max 994) vs. Fontibón train = 18.88 µg/m³
  (std 10.84, max 100.8) — la media difiere ~5.2 veces y la desviación estándar ~8.5 veces.
  Distancia de Wasserstein en unidades crudas: pm25=80.16, temp=8.68, dew=12.20; tras
  estandarizar (forma de la distribución, sin el nivel/escala): pm25=0.147, temp=0.223,
  dew=0.201 — las formas estandarizadas están bastante más cerca entre sí que los niveles
  crudos. Esto **motiva** explorar la transferencia como hipótesis razonable, pero esta métrica
  compara solo la forma de la distribución marginal, no la dinámica temporal (autocorrelación,
  secuencia); por sí sola **no demuestra** que Beijing y Fontibón compartan dinámicas
  temporales transferibles, ni que la transferencia vaya a funcionar.
- **A = EJECUTADO, 3/3 semillas completas** (42, 123, 2026): val RMSE 6.314 ± 0.061 µg/m³,
  por debajo de Ridge (6.545) en las tres. Ver `outputs/metrics/transfer_A_seed{42,123,2026}_summary.json`.
- **B = EJECUTADO, 2/3 semillas completas** (42, 123): val RMSE 6.834 ± 0.016 µg/m³. Seed 2026
  **PROVISIONAL**: interrumpida por un reinicio del equipo en la época 83, evaluada por
  inferencia pura sobre el checkpoint de su mejor época (74) — RMSE=6.945, excluida de la media.
- **C = EJECUTADO, 3/3 semillas completas** (42, 123, 2026): val RMSE 6.627 ± 0.040 µg/m³.
- **D = SOLO PILOT de integración** (2 épocas, `transfer_pilot_D_seed42_summary.json`, val
  RMSE=6.149, no comparable a un entrenamiento completo) — **0/3 semillas con entrenamiento
  completo**. No es obligatorio según la guía del curso (sección 7). `early_stopping.patience=15`/
  `min_delta=0.0` ya quedaron fijados (no son `null`/TBD) e idénticos en los 4 `configs/transfer_*.yaml`.
- Diagnóstico de convergencia/sobreajuste de A/B/C: ver `docs/ANALISIS_SOBREAJUSTE.md` — en el
  rango de épocas entrenado, sin evidencia de sobreajuste en ninguna configuración (gap train/val
  ≤7% relativo en las 8 corridas; val_loss queda por debajo de train_loss en 6 de 8). El
  diagnóstico se limita a los historiales y épocas observados, no es una garantía general.
- **Fontibón test**: no evaluado en ninguna etapa de Fase 2 (reservado para Fase 3).

## Regla de oro: qué test se usa y cuándo

- **Fontibón validation** (2 564 ventanas, may.–sep. 2022) es el **único** conjunto que se
  consulta durante Fase 2 para seleccionar hiperparámetros, arquitectura, estrategia de
  congelamiento, tasas de aprendizaje o el "ganador" entre A/B/C/D. Criterio: RMSE de
  validación, igual que en la Primera Entrega (Ridge: RMSE val = **6.545**).
- **Fontibón test** (2 574 ventanas, sep.–dic. 2022) **no se toca en ningún experimento de
  Fase 2**, ni siquiera para "solo mirar". El valor Ridge test RMSE = 7.063 se conserva
  únicamente como resultado histórico de Fase 1, documentado y congelado; no se recalcula ni
  se compara contra ninguna configuración profunda en esta entrega. La evaluación del modelo
  profundo definitivo sobre Fontibón test corresponde a la Fase 3 (Entrega final).
- **Beijing test (2014 completo)** es distinto: **sí pertenece** al experimento de
  replicación (R1) y se usa, al finalizar R1, para contrastar contra los resultados
  publicados del paper (RMSE=19.04, MAE=11.13, R²=0.937). No es un conjunto "reservado para
  Fase 3" — es la evidencia misma que pide el requisito 7.1 de la guía.

## 1. Matriz de trazabilidad (requisito de la guía → evidencia)

| Requisito guía (§7) | Experimento que lo responde | Archivo(s) de código (planeado) | Evidencia |
|---|---|---|---|
| **7.1** Replicación del método sobre Beijing | R1: Beijing Replication (8 variables originales, one-hot en `cbwd`), evaluado contra Beijing test 2014 | `notebooks/SegundaEntrega_DL.ipynb` §4, `src/models/stn.py`, `src/data/beijing.py` | Tabla RMSE/MAE/R² vs paper (19.04/11.13/0.937), curvas train/val, discusión de diferencias |
| **7.2** Traslado al problema local | Config A (from-scratch) sobre Fontibón, reutilizando el pipeline ya verificado (`src/data/fontibon.py`) | `notebooks/SegundaEntrega_DL.ipynb` §6-7.1, `configs/transfer_A_from_scratch.yaml` | Partición idéntica a la línea base, tests de `tests/` (ya verdes), RMSE/MAE/R² en **validación** vs Ridge (6.545) |
| **7.3** Estrategia de transferencia | R2 (Beijing-3 pretrain) + Configs B/C/D vs A como contraste explícito | `src/models/stn.py`, `src/training/transfer.py`, `configs/beijing_transfer_source_r2.yaml`, `configs/transfer_{B,C,D}_*.yaml`, `notebooks/SegundaEntrega_DL.ipynb` §3, §5, §7 | Tabla A vs B vs C vs D (media±std sobre 3 semillas, todo en **validación**), análisis de distancia de dominio, justificación textual de la estrategia elegida |
| **7.4** Dinámica del entrenamiento y comparación con línea base | Todas las configuraciones anteriores, monitoreadas con TensorBoard | `src/training/*.py`, `outputs/tensorboard/`, `outputs/figures/`, `notebooks/SegundaEntrega_DL.ipynb` §8-9 | Curvas train/val loss y RMSE, norma del gradiente, LR por época; tabla final STN vs Ridge (**val** 6.545; test 7.063 mostrado solo como referencia histórica, no recalculado) |

El único notebook académico de esta fase es `notebooks/SegundaEntrega_DL.ipynb` (ver §7 más
abajo); no existen notebooks separados por sub-tema. Estado actual del código: módulos de
datos/métricas construidos y probados (`src/data/fontibon.py`; `src/data/beijing.py`, ahora con
`prepare_beijing_r1` — split cronológico, one-hot y MinMax ajustados solo con train, ventanas
causal-seguras, `24 tests`; `src/evaluation/metrics.py`, con `evaluate_scaled_predictions` para
convertir predicciones normalizadas a µg/m³); `src/models/sparse_attention.py` implementado y
probado de forma aislada; `src/models/stn.py` (arquitectura completa, con semillas únicas por
bloque vía `derive_child_seed`) implementado y probado con CANDIDATE_V1
(`configs/stn_candidate_v1.yaml`, no final); `src/training/trainer.py` (loop de entrenamiento
reutilizable: Adam + MSE, TensorBoard, historial CSV/JSON, norma de gradiente real por batch,
early stopping configurable). Un **pilot corto** (`scripts/run_r1_pilot.py`, ≤5 épocas sobre un
subconjunto cronológico de Beijing train/validation) verificó la cadena completa de extremo a
extremo — **no es un entrenamiento real ni mide desempeño final**. Ningún entrenamiento completo
(200 épocas), ninguna comparación de arquitecturas, ningún tuning, y ninguna evaluación de
Beijing test 2014 se ha ejecutado. Fontibón no se ha usado en esta etapa. Los `configs/*.yaml`
de los experimentos R1 completo/R2/A/B/C/D eran, en este punto del proyecto, esqueletos no
ejecutados.

**Nota (estado actual, no el de este párrafo histórico)**: R1 completo, R2 completo, y A/B/C
completos (ver estado consolidado más abajo) ya se ejecutaron — este párrafo describe el estado
en el momento en que se escribió, inmediatamente después del pilot de R1, no el estado final.

## 2. Validación temporal en Beijing — **aprobado**

El paper usa 2010–2013 para entrenamiento (con 10 % aleatorio de esos 4 años como
validación) y 2014 completo como prueba. Un split aleatorio de validación mezclaría horas
temporalmente cercanas al entrenamiento — el propio informe de Primera Entrega ya señaló esto
como limitación del paper (§6.1). Para evitar fuga temporal, se aprobó:

| Partición | Periodo | Uso |
|---|---|---|
| Train | 2010-01-01 00:00 → 2013-08-31 23:00 | ajuste de pesos |
| Validation | 2013-09-01 00:00 → 2013-12-31 23:00 (últimos 4 meses de 2013) | early stopping, selección de hiperparámetros, diagnóstico |
| Test (réplica) | 2014-01-01 00:00 → 2014-12-31 23:00 | evaluación final de R1/R2, comparación con el paper — **idéntico al split del paper**, nunca tocado durante el desarrollo/selección |

**Esto es una adaptación metodológica deliberada frente al paper**, no un intento de
replicar su procedimiento de validación aleatoria. 2014 permanece completamente separado,
igual que en el paper, preservando la comparabilidad del resultado de test con
RMSE=19.04/MAE=11.13/R²=0.937. Aplica tanto a R1 (8 variables) como a R2 (3 variables,
fuente de transferencia).

## 3. Episodios de PM2.5 elevado — conteo por partición (Fontibón)

Calculado con `scripts/run_baseline_fontibon.py` (`outputs/metrics/baseline_fontibon_ridge.json`),
sobre las particiones ya verificadas:

| Partición | n ≥ 40 µg/m³ (umbral principal, **aprobado**) | n ≥ 37 µg/m³ (umbral usado en la celda del notebook, ver PHASE1_AUDIT §2.1) | n total |
|---|---|---|---|
| Train | 469 | 707 | 10 802 |
| Validation | 66 | 90 | 2 564 |
| Test | 118 | 159 | 2 574 |

Con el umbral principal (40), validación tiene 66 observaciones altas — un número modesto
frente a las 118 de test (que no se consulta en esta fase, ver regla de oro arriba).
**RMSE_high/MAE_high/Bias_high se reportarán únicamente como diagnóstico secundario**, nunca
como criterio principal de selección de arquitectura, congelamiento o hiperparámetros; el
criterio principal sigue siendo el RMSE global de **validación**.

Como referencia histórica (Fase 1, sobre test — no recalculada en Fase 2), la línea base Ridge
mostró: RMSE_high=17.87, MAE_high=15.58, Bias_high=−14.96 (subestimación; signo negativo =
predicción por debajo del valor real, convención documentada en `src/evaluation/metrics.py`).

## 4. Distancia entre dominios (Beijing vs Fontibón) — **EJECUTADO**

Ejecutado por `scripts/run_domain_distance_analysis.py` sobre TRAIN de ambos dominios (nunca
Beijing test 2014 ni Fontibón validation/test). Resultados en
`outputs/metrics/domain_distance_analysis.json` y `outputs/figures/domain_distance/`, mostrados
en `notebooks/SegundaEntrega_DL.ipynb` §3. Cubrió, sobre las tres variables armonizadas
(pm25, temp, dew):

1. Estadísticas descriptivas (media, desviación, percentiles, rango) por ciudad.
2. Distribuciones (histogramas/KDE) por variable y ciudad.
3. Autocorrelación de PM2.5 en lag 1, 24, 168 horas, por ciudad.
4. Perfil horario promedio de PM2.5 (media por hora del día), por ciudad.
5. Comparación de distribuciones tras estandarización (no solo diferencia de nivel absoluto).
6. Distancia de Wasserstein por variable (pm25, temp, dew), Beijing vs Fontibón.
7. Tabla de diferencias de media, desviación, percentiles y rango por variable.

Interpretación prevista: la hipótesis de transferencia no es que los niveles absolutos
coincidan (ya se sabe que no: PM2.5 medio Beijing ≈98.6 vs Fontibón ≈18.3 µg/m³, informado en
Primera Entrega), sino que ciertas representaciones temporales (autocorrelación, perfil
horario) puedan ser parcialmente transferibles pese a la diferencia de escala. Este análisis
cuantificará esa hipótesis antes de mirar si la transferencia efectivamente ayuda.

## 5. Experimentos de transferencia — resumen

Ver `configs/transfer_{A,B,C,D}_*.yaml` para el detalle completo de cada configuración.
Constantes obligatorias entre A/B/C/D: mismo dataset y split de Fontibón (idéntico al de
`configs/baseline_fontibon.yaml`), mismas ventanas, mismo escalado, mismo batch size, misma
loss, mismo criterio de early stopping, mismas métricas, mismo criterio de selección (RMSE de
**validación**, nunca test), y las mismas tres semillas.

| Config | Init | Encoder | Decoder | Output head | LR |
|---|---|---|---|---|---|
| A — from scratch | aleatoria | entrenable | entrenable | entrenable (nueva) | 1e-4 (todo) |
| B — feature extractor | Beijing-3 (R2) | **congelado** | **congelado** | **nueva, entrenable — única parte que actualiza pesos** | head: 1e-4 |
| C — partial fine-tuning | Beijing-3 (R2) | **congelado** | entrenable | entrenable | decoder+head: 1e-4 |
| D — full fine-tuning | Beijing-3 (R2) | entrenable | entrenable | entrenable | encoder: 1e-5; decoder+head: 1e-4 |

**Distinción B vs C** (corregida — antes aparecían casi idénticas): en B se congela **encoder
y decoder**, y solo se entrena una cabeza de salida nueva — es el caso "extractor de
características puro". En C se congela **únicamente el encoder**; el decoder sí se entrena
junto con la cabeza. B es la transferencia más conservadora de las cuatro; C ya permite que la
parte generativa (decoder) se adapte al dominio local.

Semillas (**aprobadas**): `[42, 123, 2026]` (42 ya usada en la Primera Entrega).
Reportar media ± desviación estándar por configuración; si la diferencia entre A/B/C/D es
menor que la variabilidad entre semillas de una misma configuración, se indicará
explícitamente en el informe en vez de forzar una conclusión.

## 6. El notebook único de Segunda Entrega

`notebooks/SegundaEntrega_DL.ipynb` es el único notebook académico de esta fase (reemplaza la
idea inicial de tres notebooks separados). Creado como **esqueleto narrativo** — markdown +
celdas placeholder que importan funciones de `src/`, sin entrenamiento pesado todavía:

```
0. Configuración y reproducibilidad
1. Carga y auditoría de datos        (1.1 Beijing, 1.2 Fontibón)
2. Preparación de datos              (2.1 Particiones, 2.2 Faltantes, 2.3 Escalado, 2.4 Ventanas)
3. Distancia entre dominios          (3.1 Descriptivas, 3.2 Distribuciones, 3.3 ACF, 3.4 Perfil horario, 3.5 Wasserstein)
4. Replicación STN sobre Beijing     (4.1 Decisiones, 4.2 Arquitectura, 4.3 Entrenamiento, 4.4 vs paper)
5. Beijing-3 como modelo fuente
6. Traslado a Fontibón
7. Transfer learning                 (7.1 A, 7.2 B, 7.3 C, 7.4 D)
8. Dinámica del entrenamiento        (8.1 Curvas, 8.2 Gradientes, 8.3 Variabilidad entre semillas)
9. Comparación contra Ridge
10. Conclusiones de Segunda Entrega
```

`PrimeraEntrega_DL.ipynb` permanece sin modificar en la raíz del proyecto.

## 7. Pendiente de aprobación antes de continuar

**R1 está completo y cerrado**: arquitectura, hiperparámetros, preprocesamiento, particiones y
checkpoints congelados (`outputs/metrics/R1_frozen_protocol.json`); validación de tres semillas
completa; evaluación única sobre Beijing test 2014 completa
(`outputs/metrics/R1_test_final_summary.json`). No se harán más cambios a R1.

R2 también está completo y cerrado (las tres semillas, ver arriba): sus tres checkpoints se
conservan como fuente para la transferencia, emparejados por semilla.

A, B y C ya se ejecutaron sobre Fontibón (ver tabla arriba: A 3/3, B 2/3+provisional, C 3/3).

Autorización pendiente para el siguiente paso, que requiere aprobación aparte y explícita:

1. Entrenamiento COMPLETO de Config D (0/3 semillas; solo pilot de integración ejecutado) —
   la infraestructura y los learning rates diferenciados (encoder=1e-5, decoder+head=1e-4) ya
   están implementados y probados (`src/training/transfer.py`), pero no se ha corrido un
   entrenamiento completo con ninguna semilla.
2. Reentrenamiento desde cero de B-seed2026 (quedó provisional tras un reinicio del equipo).
3. Cualquier ajuste de hiperparámetros de CANDIDATE_V1 — no se hará basándose en los
   resultados de test de R1 ni de validación de R2/A/B/C; si la transferencia lo requiere, será
   una decisión nueva y documentada aparte, no una reapertura de R1/R2/A/B/C.

Comandos exactos ya usados para el entrenamiento completo de A/B/C (y previstos, sin ejecutar
todavía, para D):

```bash
# Pilots de integracion (ejecutados, 2 epocas, 512/128 ventanas -- no son resultado final):
python scripts/run_transfer_pilot.py --config A --seed 42 --max-epochs 2 --max-train-windows 512 --max-validation-windows 128
python scripts/run_transfer_pilot.py --config B --seed 42 --max-epochs 2 --max-train-windows 512 --max-validation-windows 128
python scripts/run_transfer_pilot.py --config C --seed 42 --max-epochs 2 --max-train-windows 512 --max-validation-windows 128
python scripts/run_transfer_pilot.py --config D --seed 42 --max-epochs 2 --max-train-windows 512 --max-validation-windows 128

# Entrenamiento completo (un comando por semilla; A/B/C ya ejecutados, D pendiente):
python scripts/run_transfer_full_training.py --config A --seed 42      # ejecutado
python scripts/run_transfer_full_training.py --config A --seed 123     # ejecutado
python scripts/run_transfer_full_training.py --config A --seed 2026    # ejecutado
python scripts/run_transfer_full_training.py --config B --seed 42      # ejecutado
python scripts/run_transfer_full_training.py --config B --seed 123     # ejecutado
python scripts/run_transfer_full_training.py --config B --seed 2026    # interrumpido -- PROVISIONAL
python scripts/run_transfer_full_training.py --config C --seed 42      # ejecutado
python scripts/run_transfer_full_training.py --config C --seed 123     # ejecutado
python scripts/run_transfer_full_training.py --config C --seed 2026    # ejecutado
python scripts/run_transfer_full_training.py --config D --seed 42      # NO ejecutado -- requiere autorizacion
python scripts/run_transfer_full_training.py --config D --seed 123     # NO ejecutado -- requiere autorizacion
python scripts/run_transfer_full_training.py --config D --seed 2026    # NO ejecutado -- requiere autorizacion
```

# Transferencia de una STN para pronóstico horario de PM2.5 (Beijing → Fontibón)

Proyecto Integrador de Deep Learning — Universidad Externado de Colombia. Réplica de Zhang &
Zhang (2023), *Modeling air quality PM2.5 forecasting using deep sparse attention-based
transformer networks*, y traslado del método a la estación Fontibón (Bogotá) mediante
transferencia de aprendizaje.

## Estado actual: Fase 3 (Entrega final) — auditoría y comunicación CERRADAS

La parte experimental está congelada. No se reentrenó nada en esta fase: todo lo que sigue es inferencia
sobre los checkpoints de la Fase 2 y análisis de resultados ya guardados.

- **Modelo final:** configuración A (STN entrenada desde cero en Fontibón), semillas 42, 123 y 2026. Se
  eligió por menor RMSE medio de validación y se congeló en `outputs/metrics/F3_frozen_protocol.json`
  **antes** de evaluar Fontibón test.
- **Test** (2.574 ventanas, una sola evaluación): A RMSE 6,869 ± 0,020; Ridge 7,063; persistencia 8,232;
  C 7,274; B 7,484. En episodios de PM2,5 ≥ 40 µg/m³ (n = 118): A 17,25, Ridge 17,87, persistencia 16,66.
- **Dictamen:** la profundidad queda justificada por un margen estrecho en el error promedio, pero no para
  anticipar episodios altos. La transferencia desde Beijing no se justifica. Detalle en
  `docs/PHASE3_RESULTADOS.md`.
- **No ejecutado (decisión declarada):** la configuración D (solo piloto) y la semilla 2026 de B
  (interrumpida en la época 83).

### Orden de ejecución de la Fase 3

Ningún script entrena modelos. Requieren el CSV de Fontibón y los checkpoints de A/B/C, que no se
distribuyen (ver "Qué se excluye").

| Paso | Script | Produce |
|---|---|---|
| 1 | `scripts/freeze_f3_protocol.py` | `F3_frozen_protocol.json` (modelo final, métricas, umbrales, SHA-256 de checkpoints). Se niega a sobrescribir. |
| 2 | `scripts/run_f3_test_evaluation.py` | `F3_test_final_summary.json`, `F3_*_test_predictions.csv`, `figures/F3_test_final/`. Se niega a ejecutarse dos veces. |
| 3 | `scripts/run_f3_explainability.py` | `F3_explainability_summary.json`, `F3_explainability_arrays.npz`, `figures/F3_explainability/` |
| 4 | `scripts/run_f3_attention_matrices.py` | `F3_attention_matrices_summary.json`, `F3_attention_matrices_arrays.npz`, `figures/F3_attention/`, `test_residual_distribution.png`, `test_model_comparison.png` |
| 5 | `scripts/run_f3_failure_analysis.py` | `F3_failure_analysis_summary.json`, `figures/F3_failure_analysis/` |
| 6 | `scripts/build_f3_notebook.py` y luego `jupyter execute --inplace notebooks/EntregaFinal_DL.ipynb` | Notebook final ejecutado (solo lee los JSON/CSV anteriores) |
| 7 | Compilar `docs/informe_final/main.tex` con pdfLaTeX (Overleaf) | Informe final |

Los pasos 3 y 4 se ejecutaron dos veces y produjeron resultados idénticos.

### Pruebas

`tests/test_f3_audit.py` cubre las piezas nuevas:
- la extracción de atención reproduce el forward del modelo;
- los pesos tienen la forma esperada, son finitos y están normalizados;
- el protocolo fija el modelo A y el umbral de 40;
- la evaluación de test fue solo inferencia, sobre 2.574 ventanas;
- las predicciones y Ridge comparten fechas y valores reales.

Los tests que dependen de archivos ausentes se omiten (`skip`) en lugar de fallar.

Última ejecución real de `pytest tests/ -v`, en el proyecto completo con datos y checkpoints:
**209 passed, 0 failed, 0 skipped, 1142 warnings**. El log está en `outputs/pytest_fase3_full.txt`. Los
warnings son avisos de Keras sobre capas personalizadas sin `build()` explícito y de NumPy 2 sobre
`copy=`; ninguno es un fallo.

En el **paquete de entrega** (sin el CSV de Fontibón ni checkpoints) el resultado real, medido sobre una
copia extraída del ZIP en una carpeta limpia, es **179 passed, 0 failed, 30 skipped, 0 errors**. El log
está en `outputs/pytest_fase3_paquete.txt`.
- 29 tests se omiten porque falta el CSV institucional de Fontibón y 1 porque faltan los checkpoints.
- Cada omisión explica su motivo, y solo se activa si el recurso no existe; ninguna aserción cambió.
- El detalle por archivo está en `docs/README_REPRODUCIBILIDAD.md`.

### Qué se excluye del paquete de entrega y por qué

- `Raw_Consolidated_ECA_Complete_2022.csv` (Fontibón): archivo institucional sin enlace público
  verificable.
- `outputs/checkpoints/`: pesos entrenados, excluidos como en la Fase 2. Hay que regenerarlos o pedir acceso.
- `outputs/tensorboard/`: registros de monitoreo, regenerables al reentrenar.
- `.venv/`, los PDF del artículo y de la guía del curso (material con derechos o del curso) y los
  entregables de fases anteriores.

El CSV de Beijing (UCI, CC BY 4.0) sí se incluye. En las copias de `outputs/metrics/*.json` del paquete,
las rutas absolutas del equipo de desarrollo se reemplazaron por rutas relativas al proyecto. Los
valores no cambian, y los originales se conservan en el equipo de desarrollo.

## Fase 2 (histórico) — R1 y R2 CERRADOS; transferencia A (3/3), B (2/3 + 1 provisional)
y C (3/3) con entrenamiento completo ejecutado; D solo con pilot de integración

Ver `docs/PHASE2_PLAN.md` para el plan completo, la matriz de trazabilidad frente a los
requisitos de la guía y el estado detallado de cada experimento. Ver `docs/PHASE1_AUDIT.md` para
la auditoría de la Primera Entrega y de ambos datasets, y `docs/IMPLEMENTATION_ASSUMPTIONS.md`
para el diseño exacto de cada pieza (incluida la sección "R2 — preentrenamiento source para
transferencia").

- **R1 (réplica Beijing, 8 variables): CERRADO.** Entrenado y evaluado contra Beijing test 2014
  una sola vez, con 3 semillas (42, 123, 2026). Test RMSE medio 21.127 ± 0.400 µg/m³ (referencia
  publicada: 19.04). No se modifica ni se reentrena.
- **R2 (fuente de transferencia Beijing-3, pm25/temp/dew): COMPLETO.** Las 3 semillas entrenadas
  completas sobre Beijing train/validation. Validation RMSE medio 21.060 ± 0.085 µg/m³. Los tres
  checkpoints (`outputs/checkpoints/R2_source_seed{42,123,2026}/best.weights.h5`) se conservan
  como fuente para A/B/C/D, emparejados por semilla (nunca se elige "el mejor").
- **Pipeline Fontibón para la STN (`src/data/fontibon.py::prepare_fontibon_stn`): implementado y
  probado.** Mismas 3 variables/orden que R2, mismo split 70/15/15 ya aprobado (train=10802,
  val=2564, test=2574 — idéntico al usado por la línea base Ridge), escalador propio de
  Fontibón (nunca el de Beijing).
- **Infraestructura de transferencia A/B/C/D (`src/training/transfer.py`): implementada, probada
  y EJECUTADA (A/B/C).** Carga selectiva de encoder/decoder desde un checkpoint R2 (la cabeza de
  salida nunca se transfiere), congelamiento por submódulo con modo de inferencia forzado
  (Dropout/SparseAttention deterministas) en las partes congeladas, y tasas de aprendizaje
  diferenciadas para Config D (encoder=1e-5, decoder+cabeza=1e-4).
- **Análisis de distancia de dominio (Beijing vs. Fontibón, solo train): ejecutado.** Ver
  `outputs/metrics/domain_distance_analysis.json` y `outputs/figures/domain_distance/`.
- **Entrenamiento completo sobre Fontibón validation (2564 ventanas, test NUNCA usado):**
  - **A** (sin transferencia, desde cero): **3/3 semillas completas** — val RMSE 6.314 ± 0.061
    µg/m³, por DEBAJO de Ridge (6.545) en las tres.
  - **B** (encoder+decoder congelados): **2/3 completas** (42, 123 — val RMSE 6.834 ± 0.016);
    seed 2026 quedó **provisional** (interrumpida por un reinicio del equipo en la época 83,
    evaluada por inferencia pura sobre su checkpoint de la mejor época — RMSE=6.945, no entra en
    la media).
  - **C** (solo encoder congelado): **3/3 semillas completas** — val RMSE 6.627 ± 0.040.
  - **D** (fine-tuning completo, LR diferenciada): solo **pilot de integración** (2 épocas, no es
    resultado final); sin entrenamiento completo en ninguna semilla — no es obligatorio según la
    guía del curso. Ver `docs/ANALISIS_SOBREAJUSTE.md` para el diagnóstico de convergencia de
    A/B/C (en el rango de épocas entrenado, sin evidencia de sobreajuste).

## Estructura

```
data/                   raw/ y processed/ (vacías por ahora; los CSV originales se leen
                         directamente desde la raíz del proyecto, ver docs/PHASE1_AUDIT.md
                         sobre por qué no se movieron)
docs/                    PHASE1_AUDIT.md, IMPLEMENTATION_ASSUMPTIONS.md, PHASE2_PLAN.md
notebooks/               SegundaEntrega_DL.ipynb — único notebook de Fase 2; el original
                         PrimeraEntrega_DL.ipynb permanece en la raíz, sin modificar
src/
  data/                  common.py (contrato compartido PreparedSplit); fontibon.py (pipeline
                         Ridge ya auditado + prepare_fontibon_stn para la STN); beijing.py
                         (prepare_beijing_r1 y prepare_beijing_r2); transfer_features.py
                         (orden/nombres canónicos de las 3 variables compartidas)
  evaluation/            metrics.py — RMSE/MAE/R2 + métricas de episodios altos; plots.py
  models/                sparse_attention.py — capa SparseAttention, probada de forma aislada;
                         stn.py — arquitectura STN completa (CONGELADA, no se modifica);
                         transfer_compat.py — auditoría de compatibilidad de pesos source/target
  training/              trainer.py — loop de entrenamiento custom (R1/R2, ya usado); transfer.py
                         — infraestructura A/B/C/D (TransferSTNModel, DifferentialLRTrainer),
                         construida, probada y EJECUTADA (A/B/C completos; D solo pilot)
  utils/                 paths.py — detección de raíz del proyecto; seeding.py
configs/                 baseline_fontibon.yaml (ejecutado) + stn_candidate_v1.yaml (arquitectura
                         congelada) + beijing_r1_development.yaml (ejecutado, R1 cerrado) +
                         beijing_transfer_source_r2.yaml (ejecutado, R2 completo) +
                         transfer_A/B/C_*.yaml (EJECUTADOS: A 3/3, B 2/3+1 provisional, C 3/3) +
                         transfer_D_full_finetune.yaml (solo pilot de integración, sin
                         entrenamiento completo)
outputs/                 checkpoints/, figures/, metrics/, tensorboard/ — incluye los resultados
                         completos de R1, R2 y A/B/C (3 semillas cada uno, salvo B-2026
                         provisional), el pilot de D, y el análisis de distancia de dominio
tests/                   pytest tests/ -v — particiones, fuga de scaler, ventanas, fuga de
                         objetivo, shapes, SparseAttention, STN, pipeline R1, pipeline R2,
                         pipeline Fontibón-STN, infraestructura de transferencia. **Todos en
                         verde en el proyecto original** (con ambos CSV y los checkpoints de R2
                         disponibles). **En el paquete de entrega** (`SegundaEntrega_Codigo_Reproducible.zip`,
                         sin el CSV de Fontibón ni checkpoints `.h5`) el resultado real medido es
                         `2 failed, 172 passed, 1 skipped, 26 errors` — ver la tabla detallada en
                         `docs/README_REPRODUCIBILIDAD.md` dentro de ese ZIP.
scripts/                 smoke_test_stn.py, run_r1_*.py, compare_r1_seeds.py, run_r2_*.py,
                         compare_r2_source_seeds.py, run_domain_distance_analysis.py (ejecutado,
                         descriptivo), run_baseline_fontibon.py (referencia histórica de la
                         Primera Entrega — NO recomendado para ejecutarse en esta fase, evalúa
                         Fontibón test)
```

## Cómo ejecutar

El entorno de referencia utilizó Python 3.12.10. Las dependencias principales se encuentran fijadas
en `requirements.txt` (verificado con un venv local en `.venv/`; no versionar esa carpeta).

```bash
pip install -r requirements.txt
python scripts/smoke_test_stn.py                  # verifica la arquitectura STN con datos sintéticos
python scripts/run_domain_distance_analysis.py    # analisis descriptivo Beijing vs Fontibon (train only)
pytest tests/ -v                                  # suite completa: particiones, leakage, shapes,
                                                   # SparseAttention, STN, R1, R2, Fontibon-STN, transfer infra
                                                   # (resultado depende de si el CSV de Fontibon y
                                                   # los checkpoints de R2 estan disponibles, ver arriba)
```

`scripts/run_baseline_fontibon.py` **no se incluye en los comandos recomendados para esta fase**:
reproduce la línea base Ridge de la Primera Entrega evaluando directamente sobre Fontibón test
(RMSE test=7.063), lo cual ya se hizo una sola vez y queda como referencia histórica congelada.
Volver a ejecutarlo en Fase 2 violaría la regla de no tocar Fontibón test.

Los experimentos pesados (`run_r1_development.py`, `run_r2_source_training.py`,
`run_transfer_full_training.py`) ya no se listan aquí como comandos de "cómo ejecutar" de
rutina: R1, R2 y las transferencias A/B/C ya están ejecutados y cerrados/completos y no deben
relanzarse (ver `docs/PHASE2_PLAN.md` para los resultados y los comandos exactos ya usados, uno
por semilla). Solo Config D sigue sin entrenamiento completo y requiere autorización explícita
aparte antes de correrse.

Todas las rutas se resuelven de forma relativa a la raíz del proyecto
(`src/utils/paths.py`), detectada automáticamente — el código corre igual desde VS Code que
desde una sesión de Colab que haya clonado/copiado el repo completo.

## Datos

Los archivos originales (`PRSA_data_2010.1.1-2014.12.31.csv`, Beijing; y
`Raw_Consolidated_ECA_Complete_2022.csv`, Fontibón) se mantienen sin modificar en la raíz del
proyecto como evidencia de la Fase 1. No se descargan copias adicionales desde internet.

**Nota sobre el paquete de entrega** (`ENTREGA_FINAL_SEGUNDA_ENTREGA/SegundaEntrega_Codigo_Reproducible.zip`):
ese ZIP incluye únicamente el CSV **público** de Beijing (`PRSA_data_2010.1.1-2014.12.31.csv`).
El CSV institucional de Fontibón (`Raw_Consolidated_ECA_Complete_2022.csv`) **no se incluye** en
ese paquete — ver `docs/README_REPRODUCIBILIDAD.md` dentro del ZIP para el detalle de qué
pruebas y experimentos pueden ejecutarse sin él y cuáles no.

## Entregables finales
Las versiones finales del informe y de la sustentación están en `docs/final_deliverables/`.

## Archivos no distribuidos
Por privacidad/tamaño no se incluyen el CSV institucional de Fontibón, checkpoints de entrenamiento ni TensorBoard. Los tests que dependen exclusivamente de esos recursos se omiten de forma explícita cuando no están disponibles.

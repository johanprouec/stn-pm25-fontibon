# Paquete reproducible — Entrega final (Fase 3), Deep Learning

Universidad Externado de Colombia. Estudiante: Johan Velandia.
Profesor: Wilmer Darío Pineda Ríos. Asignatura: Deep Learning. Entrega final, 2026.

Este archivo describe el contenido real de `EntregaFinal_Codigo_Reproducible.zip`. Sustituye a la
versión de la Segunda Entrega.

## Qué incluye el paquete

```
README.md, requirements.txt
PRSA_data_2010.1.1-2014.12.31.csv     CSV de Beijing (público, UCI, CC BY 4.0)
src/                                   código fuente (datos, modelos, entrenamiento, evaluación, utilidades)
scripts/                               scripts de las tres fases, incluidos los de la Fase 3
configs/                               YAML de todos los experimentos (R1, R2, A/B/C/D)
tests/                                 suite completa, incluida tests/test_f3_audit.py
notebooks/EntregaFinal_DL.ipynb        notebook final, ejecutado (lee los JSON/CSV, no recalcula)
notebooks/SegundaEntrega_DL.ipynb      notebook de la Segunda Entrega, ejecutado
outputs/metrics/                       todos los JSON/CSV/NPZ de resultados (Fases 2 y 3)
outputs/figures/                       todas las figuras PNG (Fases 2 y 3)
outputs/pytest_fase3_full.txt          log de pytest en el proyecto completo
outputs/pytest_fase3_paquete.txt       log de pytest en una copia extraída de este paquete
docs/                                  PHASE1_AUDIT, PHASE2_PLAN, PHASE3_RESULTADOS, IMPLEMENTATION_ASSUMPTIONS,
                                       ANALISIS_SOBREAJUSTE, este archivo e informe_final/ (main.tex + figuras)
data/, outputs/checkpoints/, outputs/tensorboard/   carpetas vacías con .gitkeep
```

`src/utils/paths.py` encuentra la raíz del proyecto buscando `requirements.txt`, así que el paquete
funciona descomprimido en cualquier carpeta sin configurar rutas.

## Qué NO incluye y por qué

| Recurso | Motivo |
|---|---|
| `Raw_Consolidated_ECA_Complete_2022.csv` (Fontibón) | Archivo institucional sin autorización confirmada de redistribución ni enlace público verificable. Alfonso Albarracín et al. (2023) se cita como antecedente de datos de Fontibón, no como procedencia demostrada de este archivo. |
| `outputs/checkpoints/**/*.h5` | Pesos entrenados (R1, R2, A/B/C/D), excluidos como en la Fase 2. Sus SHA-256 de A/B/C están en `outputs/metrics/F3_frozen_protocol.json`, para verificar integridad si se obtienen. |
| `outputs/tensorboard/**` | Registros de monitoreo, regenerables al reentrenar; los resultados ya están en los JSON. |
| `.venv/`, cachés | Entorno local; se recrea con `requirements.txt`. |

En las copias de `outputs/metrics/*.json` del paquete, las rutas absolutas del equipo del autor se
reemplazaron por rutas relativas al proyecto. Ningún valor cambia y los originales se conservan en el
equipo del autor.

## Cómo ejecutar las pruebas

El entorno de referencia utilizó Python 3.12.10. Las dependencias principales se encuentran fijadas en
`requirements.txt`.

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Resultados reales medidos, no estimados:

| Entorno | passed | failed | skipped | errors | Log |
|---|---|---|---|---|---|
| Proyecto completo (con CSV de Fontibón y checkpoints) | 209 | 0 | 0 | 0 | `outputs/pytest_fase3_full.txt` |
| Este paquete, extraído en una carpeta limpia | 179 | 0 | 30 | 0 | `outputs/pytest_fase3_paquete.txt` |

En ambos casos aparecen 1.142 warnings: avisos de Keras porque la capa `SparseAttention` no tiene `build()`
explícito, y avisos de NumPy 2 sobre `copy=`. No son fallos.

**Tests que se omiten en el paquete** y su motivo:

| Tests | Recurso ausente | Mensaje |
|---|---|---|
| `test_fontibon_stn_pipeline.py` (13), `test_partitions.py` (4), `test_scaler_leakage.py` (3), `test_shapes.py` (3 de 6), `test_target_leakage.py` (3), `test_windows_no_cross_partition.py` (1 de 4), `test_f3_audit.py::test_fontibon_test_has_2574_windows` (1) | CSV de Fontibón | `Skipped: Fontibón institutional dataset is not distributed with the reproducible package.` |
| `test_beijing_r2_pipeline.py::test_beijing_r2_order_matches_fontibon_order` (1) | CSV de Fontibón | Mensaje propio desde la Fase 2. El mismo contrato se verifica siempre con un CSV sintético en `test_fontibon_feature_order_contract_synthetic`. |
| `test_transfer_infrastructure.py::test_r2_source_checkpoints_still_on_disk_and_unmodified_count` (1) | Checkpoints de R2 | `Skipped: training checkpoint is intentionally excluded from the submission package.` Solo se omite si no existe **ninguno** de los tres; si faltara solo uno, el test falla. |

Las omisiones solo se activan cuando el archivo no existe. Con los recursos presentes, todos esos tests se
ejecutan con las mismas aserciones, como muestra la primera fila de la tabla.

## Qué se puede reproducir con este paquete

- **Sin recursos externos:**
  - toda la suite de tests salvo los 30 omitidos;
  - la parte Beijing completa (R1 y R2), que solo necesita el CSV incluido;
  - la lectura y verificación de todos los resultados guardados: el notebook final, los JSON y las figuras.
- **Con el CSV de Fontibón:** el preprocesamiento, Ridge y el entrenamiento de A/B/C, que no se recomienda
  repetir (ver abajo). Además pasan los 29 tests de Fontibón.
- **Con el CSV y los checkpoints:** también los scripts de auditoría de la Fase 3 (inferencia), cuyos
  resultados deben coincidir con los JSON guardados.

## Orden de ejecución de la Fase 3 (no entrena modelos)

| Paso | Comando | Nota |
|---|---|---|
| 1 | `python scripts/freeze_f3_protocol.py` | Se niega a sobrescribir el protocolo existente. |
| 2 | `python scripts/run_f3_test_evaluation.py` | Evaluación única de Fontibón test. Se niega a correr si ya existe `F3_test_final_summary.json`. **No re-ejecutar.** |
| 3 | `python scripts/run_f3_explainability.py` | Atención, gradientes integrados, casos sintéticos y ablaciones. |
| 4 | `python scripts/run_f3_attention_matrices.py` | Matrices consulta × clave, cabezas, normal/alto, residuos y comparación final. |
| 5 | `python scripts/run_f3_failure_analysis.py` | Fallas por subgrupo, calibración y fuera de dominio. |
| 6 | `python scripts/build_f3_notebook.py` y `jupyter execute --inplace notebooks/EntregaFinal_DL.ipynb` | Regenera y ejecuta el notebook final. |
| 7 | Compilar `docs/informe_final/main.tex` con pdfLaTeX (por ejemplo en Overleaf) | El ZIP del informe contiene `main.tex` y las 11 figuras. |

Los pasos 3 y 4 se ejecutaron dos veces en el proyecto original y produjeron resultados idénticos.

## Cómo reproducir sin arriesgar la evidencia

Los scripts de entrenamiento (`run_r1_development.py`, `run_r2_source_training.py`,
`run_transfer_full_training.py`) y los de la Fase 3 que tocan test se niegan a correr si su resultado ya
existe. Para verificar algo ya ejecutado:

1. Trabaje sobre una **copia** del proyecto, nunca sobre el original.
2. Dentro de la copia, borre solo los artefactos del experimento que quiera repetir.
3. Ejecute el script.

`run_baseline_fontibon.py` evalúa Ridge sobre Fontibón test. Se ejecutó una sola vez en la Primera Entrega y
se reproduce dentro de `run_f3_test_evaluation.py` como comprobación (RMSE 7,063). No debe usarse para tomar
decisiones.

## Estación Fontibón

El archivo institucional contiene las estaciones `FONTIBON` y `MOVIL FONTIBON`. Todo el proyecto usa
exclusivamente `STATION == "FONTIBON"`: el filtro está fijado en `src/data/fontibon.py`.

## Estado de los experimentos (congelado)

- **R1** (Beijing): cerrado; test 2014 RMSE 21,13 ± 0,40.
- **R2** (fuente de transferencia): completo; validación RMSE 21,06 ± 0,085.
- **Fontibón, validación:**
  - A: 6,314 ± 0,061 (3 semillas);
  - B: 6,834 ± 0,016 (2 semillas; la semilla 2026 se interrumpió y está excluida);
  - C: 6,627 ± 0,040 (3 semillas);
  - D: solo piloto.
- **Fontibón, test** (evaluado una sola vez, modelo final A): 6,869 ± 0,020, frente a Ridge 7,063.
  Detalle y dictamen en `docs/PHASE3_RESULTADOS.md`.

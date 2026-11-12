---
name: ANALISIS_SOBREAJUSTE
description: Diagnóstico de convergencia/sobreajuste, estabilidad de gradiente, variabilidad entre semillas y costo computacional para Config A/B/C sobre Fontibón, con evidencia real de los historiales de entrenamiento guardados
---

# Diagnóstico de entrenamiento — Config A/B/C sobre Fontibón (validación)

Basado en `outputs/metrics/transfer_{cfg}_seed{seed}_history.json` y `_summary.json` reales —
ningún número aquí es estimado o inventado.

## 1. Diferencia train loss vs. validation loss (en la época de mejor val_loss)

| Config-seed | train_loss@best | val_loss@best | gap (val−train) |
|---|---|---|---|
| A-42 | 0.00398 | 0.00402 | +0.00004 |
| A-123 | 0.00396 | 0.00389 | −0.00006 |
| A-2026 | 0.00377 | 0.00388 | +0.00011 |
| B-42 | 0.00486 | 0.00459 | −0.00027 |
| B-123 | 0.00491 | 0.00462 | −0.00029 |
| C-42 | 0.00457 | 0.00432 | −0.00024 |
| C-123 | 0.00455 | 0.00428 | −0.00026 |
| C-2026 | 0.00452 | 0.00439 | −0.00013 |

**En el rango de épocas efectivamente entrenado, los historiales no muestran evidencia de
sobreajuste** (esto no descarta que pudiera aparecer más allá del punto de parada, algo que este
experimento no exploró): el gap es del orden de 10⁻⁴-10⁻⁵ en una pérdida de ~4-5×10⁻³ (≤7%
relativo), y en **6 de 8 corridas** val_loss queda incluso por DEBAJO de train_loss (no es
anómalo: train_loss se promedia sobre más ventanas y con mayor variabilidad temporal que
validation; no indica mejor generalización, solo que ambos conjuntos están en el mismo régimen
de ajuste). Un sobreajuste típico se vería como un gap positivo creciente y significativo — eso
no se observa en ninguno de los 8 historiales analizados.

## 2. Época óptima y criterio de parada

| Config-seed | Best epoch | Stopping epoch | Gap (stop−best) |
|---|---|---|---|
| A-42 | 102 | 117 | 15 (patience agotado) |
| A-123 | 96 | 111 | 15 (patience agotado) |
| A-2026 | 110 | 125 | 15 (patience agotado) |
| B-42 | 188 | 200 | 12 (tope de max_epochs=200, NO patience) |
| B-123 | 111 | 126 | 15 (patience agotado) |
| C-42 | 59 | 74 | 15 (patience agotado) |
| C-123 | 46 | 61 | 15 (patience agotado) |
| C-2026 | 53 | 68 | 15 (patience agotado) |

En 7 de 8 corridas el criterio de parada (`patience=15, monitor=val_loss, min_delta=0.0`) se
activó correctamente. **B-seed42 es la excepción**: llegó al tope de 200 épocas sin agotar la
paciencia (solo 12 épocas sin mejora, no 15) — es decir, val_loss seguía mejorando muy
lentamente cuando se acabaron las épocas permitidas. No es un fallo del criterio, es evidencia
de que B converge más lento que A y C en al menos una semilla.

## 3. Estabilidad del gradiente

| Config-seed | Grad. inicial | Grad. final | Grad. máximo |
|---|---|---|---|
| A-42 | 4.032 | 0.150 | 65.67 |
| A-123 | 4.079 | 0.163 | 70.95 |
| A-2026 | 2.289 | 0.151 | 21.18 |
| B-42 | 0.216 | 0.122 | 1.56 |
| B-123 | 0.310 | 0.129 | 1.80 |
| C-42 | 0.250 | 0.163 | 2.18 |
| C-123 | 0.254 | 0.180 | 1.93 |
| C-2026 | 0.295 | 0.138 | 2.94 |

Patrón claro y esperado: **A arranca con gradientes 10-20x más grandes** que B/C (encoder
inicializado aleatoriamente, lejos de cualquier solución razonable) y los máximos durante el
entrenamiento son igualmente mucho mayores (21-71 vs. 1.5-2.9). B y C, al partir de pesos ya
preentrenados en Beijing (R2), arrancan con activaciones ya cercanas a una solución razonable y
el gradiente nunca necesita moverse tanto. En los tres grupos el gradiente decrece y se
estabiliza hacia el final — sin divergencia ni explosión en ningún caso.

**Qué sí y qué no dice la norma del gradiente**: esta métrica sirve para diagnosticar
estabilidad numérica del entrenamiento (explosión, divergencia, estancamiento) y para comparar
cuánto necesita moverse cada configuración desde su punto de partida. Una norma pequeña o
decreciente **no demuestra** que el modelo haya convergido a un óptimo global (podría ser un
óptimo local, o simplemente una región plana de la superficie de pérdida) **ni garantiza** una
mejor capacidad de generalización — ambas cosas requerirían evidencia adicional (por ejemplo,
desempeño en un conjunto de prueba independiente) que esta métrica, por sí sola, no provee.

## 4. Variabilidad entre semillas

- **A**: best epoch entre 96-110 (rango 14), val RMSE 6.274-6.384 (std=0.061) — variabilidad
  baja en resultado final, moderada en número de épocas necesarias.
- **B**: best epoch 111 vs. 188 (rango 77, la semilla 42 tardó casi el doble) — la configuración
  con MÁS variabilidad en dinámica de entrenamiento de las tres, aunque el resultado final
  (val RMSE 6.823 vs 6.845) es consistente.
- **C**: best epoch 46-59 (rango 13), val RMSE 6.591-6.670 (std=0.040) — la más rápida Y la más
  estable de las tres en número de épocas.

## 5. Costo computacional

| Config | Tiempo por semilla (h) | Épocas hasta best |
|---|---|---|
| A | 2.40 / 2.86 / 3.54 | 96-110 |
| B | 4.40 / 3.42 (solo 2 semillas) | 111-188 |
| C | 1.46 / 1.19 / 1.91 | 46-59 |

**C es la configuración más barata por un margen amplio** (~1.5h promedio vs. ~2.9h de A y
~3.9h de B) — congelar el encoder no solo reduce qué se actualiza, también acelera la
convergencia porque el decoder parte ya cerca de una buena representación del encoder fijo.
B, pese a partir también de pesos preentrenados, es la más costosa: congelar encoder Y decoder
deja muy poca capacidad de ajuste (solo la cabeza), lo que aparentemente obliga a más épocas
para encontrar el mejor punto de esa cabeza.

## Conclusión del diagnóstico

En el rango de épocas observado, ninguna de las tres configuraciones muestra evidencia de
sobreajuste — el diagnóstico se limita a los historiales y épocas efectivamente entrenados, no
a una garantía general. La diferencia más informativa no está
en train vs. validation (ambos coinciden de cerca en las tres), sino en la **dinámica de
convergencia**: A necesita más gradiente y más variabilidad entre semillas porque aprende todo
desde cero; C converge rápido y barato gracias al encoder congelado pero ya entrenado; B es la
más costosa en tiempo pese a partir también de pesos preentrenados, posiblemente porque congelar
ambos módulos (encoder y decoder) deja a la cabeza de salida con muy poca flexibilidad para
compensar el mismatch de dominio Beijing→Fontibón.

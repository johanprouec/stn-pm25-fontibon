# Fase 3 — Resultados, dictamen y limitaciones

Documento de referencia de la Entrega final. Todas las cifras provienen de
archivos ya calculados en `outputs/metrics/`:

| Archivo | Contenido |
|---|---|
| `F3_frozen_protocol.json` | Protocolo congelado **antes** de evaluar Fontibón test |
| `F3_test_final_summary.json` | Evaluación única sobre test (A, B, C, Ridge, persistencia) |
| `F3_explainability_summary.json` | Atención, gradientes integrados, atajo, casos sintéticos, ablaciones |
| `F3_failure_analysis_summary.json` | Fallas por subgrupo, calibración y fuera de dominio |

Scripts, en orden de ejecución: `scripts/freeze_f3_protocol.py` →
`scripts/run_f3_test_evaluation.py` → `scripts/run_f3_explainability.py` →
`scripts/run_f3_failure_analysis.py`. Ninguno entrena modelos.

---

## 8.1 Modelo final y consolidación del traslado

**Selección.** El modelo final es la configuración **A** (STN entrenada desde cero en
Fontibón). Se eligió con el criterio fijado en la Primera Entrega, el menor RMSE medio
de validación (A 6,314; C 6,627; B 6,834 µg/m³), y se congeló por escrito antes de abrir
test. A obtuvo el menor RMSE medio de validación y mostró baja variabilidad entre semillas
(DE = 0,061 en validación y 0,020 en test). Aunque C presentó menor dispersión (0,040), su
RMSE fue superior; B mostró una dispersión menor (0,016), pero solo contó con dos corridas
completas. Por ello, el criterio principal predefinido de RMSE favoreció a A. C tardó menos en entrenar (1,2–1,9 h frente a 2,4–3,5 h),
pero eso no compensa su mayor error.

**Integridad.** Antes de tocar test, los ocho checkpoints reprodujeron exactamente su RMSE
de validación de la Fase 2 (diferencia 0,0) y conservaron su SHA-256. Ridge reprodujo
su RMSE de test de la Fase 1 (7,063).

**Test** (2.574 ventanas, 14-sep-2022 a 31-dic-2022):

| Modelo | RMSE | MAE | R² | RMSE ≥ 40 µg/m³ (n = 118) | Sesgo ≥ 40 |
|---|---|---|---|---|---|
| **A** (3 semillas) | **6,869 ± 0,020** | 4,969 ± 0,037 | 0,616 ± 0,002 | 17,25 ± 0,23 | −13,95 |
| C (3 semillas) | 7,274 ± 0,038 | 5,294 | 0,569 | 18,77 | −15,73 |
| B (2 semillas) | 7,484 ± 0,051 | 5,429 | 0,544 | 19,52 | −16,70 |
| Ridge | 7,063 | 5,115 | 0,593 | 17,87 | −14,96 |
| Persistencia | 8,232 | 5,843 | 0,448 | 16,66 | −9,80 |

A reduce el RMSE de Ridge en 0,19 µg/m³ (2,7 %) en las tres semillas. El intervalo
bootstrap del 95 % (bloques móviles de 24 h) excluye el cero en las tres: [−0,28; −0,08],
[−0,33; −0,11] y [−0,34; −0,05]. La ganancia es pequeña: A acierta mejor que Ridge en
solo el 51–53 % de las ventanas.

**Contraste con el artículo.**
- *Se conservó:* la arquitectura, la ventana de 24 h, el horizonte de 1 h, la pérdida MSE,
  Adam con tasa 10⁻⁴, el escalado MinMax y las métricas. En Beijing (R1) se obtuvo RMSE
  21,13 ± 0,40 frente a 19,04 del artículo.
- *Cambió:* la profundidad (red reducida a 2 encoders y 1 decoder), las variables (3 en vez
  de 8) y la validación (cronológica en vez de aleatoria).
- *En el dominio local:* el RMSE de Fontibón (6,87) es menor que el de Beijing porque los
  niveles son mucho más bajos, pero el R² cae de 0,95 a 0,62. Fontibón es menos
  persistente (autocorrelación a 1 h de 0,74 frente a 0,96) y su variabilidad hora a hora
  es más difícil de anticipar con solo PM2,5, temperatura y punto de rocío.
- *Taizhou:* la segunda serie del artículo, de niveles comparables a los de Fontibón,
  reporta RMSE 5,79 y R² 0,924. No es una referencia controlada (otra ciudad, más
  variables, validación aleatoria), pero sugiere que la brecha de R² se debe más a la
  información disponible que a la arquitectura.

**Transferencia.** No fue útil. B y C quedan por encima de A en cada semilla emparejada y
por encima de Ridge. Por tanto, el criterio de la Primera Entrega para considerar útil la
transferencia no se cumple.

## 8.2 Auditoría mediante explicabilidad

**Verificación.** Los mapas de atención se extrajeron con un *forward* manual que
reproduce la predicción del modelo con una diferencia máxima de 2×10⁻⁷. Los gradientes
integrados cumplen la propiedad de completitud con un error medio de alrededor del 5 %.

**Qué usa el modelo** (gradientes integrados, 518 ventanas de test, 3 semillas):
- El 70 % de la atribución corresponde a las últimas 6 horas, y el 24 % solo al PM2,5 de la
  última hora.
- Por variable: PM2,5 46 %, temperatura 27 %, punto de rocío 27 %. En episodios altos, el
  PM2,5 sube al 58 %.
- Ablaciones sobre test:
  - neutralizar la meteorología: el RMSE sube +0,69;
  - permutar el orden de las 24 horas: +4,30;
  - aplanar la historia de PM2,5 en su último valor: +0,76.
- El modelo usa la meteorología y el orden temporal; no se limita a mirar un solo valor.

**Casos sintéticos de comportamiento esperado.**
- Las series constantes se reproducen: 10 → 9,9; 20 → 20,5; 40 → 38,8.
- Un pico de 60 µg/m³ ocurrido hace 12 h se ignora (15,6 sobre una base de 15).
- Los cambios recientes se amortiguan: un escalón 15→35 se predice en 28,5, una rampa
  15→45 en 32,8 y un pico de 60 en la última hora en 30,6.

**Atajo identificado.** El modelo se comporta como una **persistencia suavizada con
regresión a la media**:
- La correlación de su predicción con la última hora observada (0,90) es mayor que con el
  valor real que debe predecir (0,785).
- Solo capta el 24 % de cada subida y el 40 % de cada bajada.

En las comprobaciones realizadas no se encontró evidencia de que la imputación o el mes
expliquen este patrón: las ventanas con imputación tienen un error similar. Es una estrategia conservadora que minimiza
el MSE promedio a costa de los cambios bruscos.

**Matrices de atención consulta × clave** (`scripts/run_f3_attention_matrices.py`,
`F3_attention_matrices_summary.json`, figuras en `outputs/figures/F3_attention/`). Se usaron
los pesos reales que devuelve `SparseAttention.call(..., return_attention=True)` en la
primera capa del encoder (`enc0_long1`): tienen forma (ventanas, 4 cabezas, 12 consultas
seleccionadas, 24 claves), son finitos y cada fila suma 1. Se extrajeron con
`forward_with_attention()`, que reproduce la predicción con una diferencia de 2×10⁻⁷ o
menos y coincide con las predicciones guardadas (2×10⁻⁵ o menos).

- **Ejemplos con la semilla 42**, elegidos con reglas fijas que no dependen del error:
  - *Episodio normal* (17-sep 08:00): real 16,1, predicción 26,5.
  - *Episodio alto* (25-oct 14:00): real 47,5, predicción 18,0.

  En las cuatro cabezas aparecen **franjas verticales**: casi todas las consultas
  seleccionadas atienden las mismas claves. La cabeza 1 atiende las horas −20/−19 y la
  última hora; la cabeza 3, la última hora; la cabeza 4, las horas −21 a −18; la cabeza 2
  reparte la atención casi uniformemente.
- **Cabezas** (semilla 42, todo el test). Se comportan de forma distinta:
  - la cabeza 2 es casi uniforme (entropía normalizada 0,98);
  - las cabezas 1 y 3 son más concentradas (0,74 y 0,80);
  - la cabeza 4 mira horas antiguas (14,2 h atrás en promedio, 10 % de masa en las últimas
    6 h).

  No se atribuye a ninguna cabeza un significado físico.
- **Casos sintéticos** (constante 20, rampa 15→45 y pico de 60 en la última hora):
  - las predicciones cambian mucho (20,7, 32,7 y 30,6);
  - la distribución de la atención casi no cambia: masa en la última hora de 0,11, 0,12 y
    0,14, y en las últimas 6 h de 0,28, 0,29 y 0,31.

  En estos casos, la atención **parece estar dominada por la posición** más que por el contenido; lo que cambia la
  predicción son los valores atendidos y las capas siguientes. Los casos no tienen valor
  real, solo un criterio de razonabilidad.
- **Normal (< 40) frente a alto (≥ 40)**, en las 3 semillas:
  - no hay una diferencia reproducible en la masa de las últimas 6 h (alto − normal: −0,001,
    +0,073 y −0,017) ni en la posición media de las claves (+0,07, −1,02 y +0,25 h);
  - la entropía es menor en episodios altos en las tres semillas, pero la diferencia es
    mínima (−0,003 a −0,022);
  - **la atención no cambia de forma apreciable durante los episodios.**

**Atención: no es una explicación estable.** Con métricas y gradientes casi idénticos, las
tres semillas distribuyen la atención de forma muy distinta (masa en las últimas 6 h de
0,17, 0,28 y 0,41). La posición que se predice participa en la autoatención del decoder en
un porcentaje de ventanas que va del 77 % al 100 % según la semilla.

**Lectura integrada de las tres fuentes.**
- *La atención* indica **qué posiciones reciben peso**: un patrón casi fijo por posición,
  distinto en cada semilla e igual en episodios normales y altos.
- *Los gradientes integrados* indican **qué entradas afectan la salida**: las últimas 6 horas
  y la meteorología, de forma estable entre semillas.
- *Las ablaciones* indican **qué información necesita el modelo**: el orden temporal y la
  meteorología.

Juntas, sostienen que el modelo usa señal temporal y meteorológica pertinente, pero con un
fuerte comportamiento de persistencia suavizada y regresión hacia la media. Los mapas de
atención no prueban esto por sí solos ni se presentan como explicación causal.

## 8.3 Análisis de fallas

| PM2,5 real | n | RMSE A | Sesgo A | RMSE Ridge | RMSE persistencia |
|---|---|---|---|---|---|
| < 10 | 556 | 5,07 | +3,41 | 5,71 | 5,36 |
| 10–20 | 1.018 | 5,33 | +1,79 | 5,56 | 7,31 |
| 20–30 | 640 | 5,81 | −1,63 | 5,87 | 8,35 |
| 30–40 | 242 | 8,95 | −5,34 | 8,89 | 10,49 |
| ≥ 40 | 118 | 17,18 | −13,95 | 17,87 | 16,66 |

**Patrones de falla.**
- **Regresión a la media:** el modelo sobreestima cuando el aire está limpio y subestima los
  picos. Ridge presenta el mismo patrón, lo que sugiere una limitación compartida por la
  información disponible (tres variables de una sola estación) y no exclusiva de la STN.
- **Subidas:** cuando el PM2,5 sube 5 µg/m³ o más en una hora (n = 555), el RMSE es 10,5 y
  el sesgo −8,2. En horas estables el RMSE es 3,9.
- **Peores errores:** de los 20 peores, el 90 % son subestimaciones, el 90 % ocurren en
  subidas y el 90 % en episodios de 40 µg/m³ o más. Se concentran entre las 7 y las 9 a. m.
  y entre las 0 y la 1 a. m.
- **Errores grandes en general:** en el 10 % de errores más grandes, el 55 % son subidas,
  frente al 22 % en todo el test.
- **Imputación y mes:** las ventanas imputadas tienen un error similar (6,97 frente a 6,78)
  y no se observó una variación importante del error entre los meses del conjunto evaluado
  (6,4–7,1 entre septiembre y diciembre).

**Consecuencia en el contexto de aplicación.** El modelo falla al inicio de los episodios
de contaminación, que es justamente cuando una alerta tendría valor. Como herramienta de
alerta temprana, subestimaría el riesgo.

**Calibración.**
- Los intervalos conformales construidos con los residuos de validación cubren casi lo
  prometido en conjunto: 77–78 % para un nominal de 80 % y 87 % para 90 %.
- Por nivel, cubren de más en aire limpio (86 % y 93 % por debajo de 10 µg/m³) y colapsan
  en episodios altos: 22 % y 33 % a partir de 40 µg/m³. El error crece con el nivel, y un
  intervalo de ancho fijo no lo refleja.
- La dispersión entre semillas es pequeña (0,9 µg/m³) y casi no se relaciona con el error
  (correlación 0,20), así que no sirve como medida de confianza.

**Fuera de dominio.**
- El error crece con la novedad de la ventana (distancia a la ventana de entrenamiento más
  parecida): RMSE 5,5 en el cuartil más típico y 7,7 en el más novedoso. A supera a Ridge
  en los cuatro cuartiles.
- Ningún valor real de test supera el máximo de entrenamiento (79,8 frente a 100,8), pero
  22 ventanas tienen alguna entrada fuera del rango de train.
- Con 24 h constantes de 100, 150 y 200 µg/m³, el modelo predice 69, 70 y 71: **se satura
  hacia 70 µg/m³** y las semillas divergen (63–87). No es apto para episodios extremos.

## 8.4 Dictamen sobre la profundidad

Criterio fijado en la Primera Entrega: *la profundidad se justifica si la STN logra una
reducción consistente del RMSE frente a Ridge y mejora el comportamiento en episodios de
concentración elevada.*

1. **Reducción consistente del RMSE: se cumple.** Las tres semillas quedan por debajo de
   7,063 y los intervalos bootstrap excluyen el cero. La magnitud es pequeña: 0,19 µg/m³, un
   2,7 %.
2. **Mejora en episodios altos: se cumple solo formalmente.** En test, A obtiene 17,25
   frente a 17,87 de Ridge, pero en validación no hubo mejora (17,88 frente a 17,85). La
   persistencia es mejor que ambos en los picos (16,66), y A sigue subestimándolos en unos
   14 µg/m³.
3. **Costo.** A tiene 99.425 parámetros, tarda entre 2,4 y 3,5 h en entrenar en CPU y unos
   14 s en predecir el test. Ridge entrena en segundos y predice en 0,002 s.
4. **Datos.** Con unas 10.800 ventanas de entrenamiento, la red no superó de forma
   apreciable a un modelo lineal con la misma información. La transferencia desde Beijing,
   el mecanismo pensado para compensar la escasez de datos, empeoró el resultado.
5. **Auditoría.** La red aprendió una señal física razonable, pero su estrategia es muy
   cercana a una persistencia suavizada, algo que un modelo lineal ya captura.

**Dictamen.** Con los criterios pre-registrados, la arquitectura profunda queda
**justificada por un margen estrecho** en el error promedio. **No se justifica para el
objetivo que motivó el proyecto**, que era mejorar la anticipación de episodios altos: la
mejora en picos no se sostiene entre validación y test y es inferior a la de la
persistencia. Si el costo importa, **Ridge sigue siendo una opción razonable**, porque la
STN mejora poco a cambio de mucho más cómputo. La transferencia desde Beijing **no se
justifica**.

## 8.5 Limitaciones y proyección

**Limitaciones.**
- *Datos:* una sola estación y dos años de registros (2021–2022). El test cubre solo de
  septiembre a diciembre de 2022, una estación del año. No hay viento comparable entre
  dominios, y la correspondencia del CSV institucional con una fuente pública no pudo
  verificarse.
- *Variable derivada:* el punto de rocío de Fontibón no se midió directamente; se calculó
  a partir de temperatura y humedad relativa con la fórmula de Magnus. El modelo depende
  así de una variable derivada que acumula el error de medición de ambas entradas y el de
  la aproximación. En Beijing, en cambio, el punto de rocío es medido.
- *Modelo:* STN reducida (2 encoders y 1 decoder) sin búsqueda de hiperparámetros, y un solo
  horizonte (t+1). La configuración D no se entrenó por completo y B tiene solo 2 semillas
  (B2026 se interrumpió); ambas son decisiones declaradas, no omisiones.
- *Transferencia:* el supuesto de que Beijing y Fontibón comparten dinámicas temporales no
  se sostuvo (autocorrelación y perfiles horarios distintos). La semejanza de distribución
  tras estandarizar no implicaba dinámicas transferibles.
- *Evaluación:*
  - tres semillas no permiten una inferencia formal;
  - el bootstrap por bloques es descriptivo;
  - los intervalos conformales se calibraron con un período (mayo–septiembre) distinto del
    de test;
  - los gradientes integrados dependen de la línea base elegida (media de train);
  - los casos sintéticos fijan la meteorología en su media, así que también son entradas
    poco realistas.

**Proyección.**
- Incorporar variables con poder explicativo para los episodios: viento, altura de capa de
  mezcla y estaciones vecinas de la red de Bogotá.
- Probar pérdidas que penalicen la subestimación (pérdida cuantílica o ponderada por nivel)
  e intervalos dependientes del nivel (conformal condicional).
- Usar transferencia desde dominios más cercanos (otras estaciones de Bogotá o ciudades
  andinas) en lugar de Beijing.

**Lo que exigiría llevarlo a producción (MLOps).**

No se implementa un despliegue; esto es lo que exigiría llevar el modelo a un entorno
productivo:

- *Versionamiento:* datos (hash del CSV y de las particiones), configuración YAML, código y
  checkpoints con SHA-256, como ya se hace en el protocolo de esta fase.
- *Monitoreo:* RMSE y sesgo móviles por nivel de concentración, cobertura de los intervalos y
  frecuencia de subestimaciones en episodios de 40 µg/m³ o más.
- *Deriva de datos (data drift):* cambio en la distribución de las entradas. Se detectaría
  con la distancia de las ventanas nuevas al conjunto de entrenamiento (la métrica de
  novedad ya implementada) y comparando distribuciones con Wasserstein.
- *Deriva de concepto (concept drift):* cambio en la relación entre las entradas y el PM2,5
  futuro, por ejemplo por nuevas fuentes de emisión o cambios en el tráfico. Se detectaría
  cuando el error y el sesgo móviles empeoran aunque las entradas sigan dentro del rango
  de entrenamiento.
- *Alertas:* cuando la entrada supere el rango de entrenamiento (el modelo se satura cerca
  de 70 µg/m³), cuando la novedad pase el cuartil superior observado o cuando la cobertura
  de los intervalos caiga.
- *Modelo de respaldo:* persistencia o Ridge cuando se active una alerta o falte una
  variable.
- *Reentrenamiento:* periódico, con ventanas recientes, validado siempre con una partición
  cronológica y con un protocolo congelado como el de esta fase.
- *Esquema del pipeline:* ingesta horaria → validación de datos (huecos, rangos) →
  construcción de ventanas y escalado con el escalador versionado → predicción → registro
  de la predicción y del valor real → monitoreo (error, sesgo, cobertura, novedad) →
  alertas, modelo de respaldo y reentrenamiento programado.

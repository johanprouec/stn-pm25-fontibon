---
name: IMPLEMENTATION_ASSUMPTIONS
description: What Zhang & Zhang (2023) explicitly report vs. what we must infer or decide ourselves, before any STN code is written
---

# Supuestos de implementación — STN (Zhang & Zhang, 2023)

Basado en una lectura completa del artículo (16 páginas, texto extraído y revisado
íntegramente, incluyendo Materials and methods, Implementation details, y todas las tablas de
resultados). El artículo se apoya explícitamente en el mecanismo *ProbSparse* de **Informer**
(Zhou et al. 2021, AAAI — citado en la bibliografía del propio paper), adaptado a PM2.5; esto
es relevante porque, donde el paper es ambiguo, Informer es la referencia más razonable para
resolver la ambigüedad — pero se marca siempre como decisión nuestra, no como algo reportado
por Zhang & Zhang.

**Regla del proyecto:** ningún valor de la categoría C se presenta jamás como "reportado por
el paper" en el código, los configs (`configs/*.yaml`) ni el informe. Los configs ya
implementan esta separación con los tags `[reported]` / `[inferred]` / `[assumed]`.

## A. Parámetros explícitamente reportados por Zhang & Zhang (2023)

| Parámetro | Valor | Ubicación en el paper |
|---|---|---|
| Encoder layers | 3 | "we employ three encoders and two decoders" (Implementation details) |
| Decoder layers | 2 | ídem |
| Conv1D kernel width (self-attention distilling) | 3 | sección Encoder |
| Max-pooling stride | 2 | sección Encoder |
| Optimizador | Adam | Implementation details |
| Learning rate inicial | 1e-4 | Implementation details |
| Batch size | 32 | Implementation details |
| Épocas máximas | 200 | Implementation details |
| Loss | MSE | Implementation details |
| Normalización | min-max a [0, 1] (alcance de ajuste — sobre qué partición se calculan min/max — **no especificado**, ver nota debajo) | Implementation details |
| Ventana / lookback | 24 | Implementation details ("set to 24 for its best performance") |
| Position embedding | seno/coseno, igual al Transformer original (Vaswani et al. 2017) | sección Position embedding |
| Estructura decoder | bloque de sparse attention enmascarado + capa de atención multi-head **estándar** (no dispersa) + FFN; cada uno seguido de layer norm; residual alrededor de cada uno de los 3 sub-bloques; mapeo lineal final | sección Decoder |
| Entrada del decoder | `X_de = {X_token, X_0}`, con `X_token` = tokens de inicio y `X_0` = placeholder para la serie objetivo (estilo Informer) | sección Decoder |
| Máscara causal | dot-products enmascarados puestos a −∞ | sección Decoder |
| Métrica de sparsity | divergencia KL entre la distribución real de atención y la uniforme; aproximada por muestreo de claves | sección "Self-attention mechanism and sparse analysis", ecs. (5)-(7) |
| Complejidad resultante | O(L log L) vs O(L²) del Transformer estándar | misma sección |
| Validación en el paper | 10 % aleatorio del conjunto de entrenamiento (**no cronológico**) | sección "Data description" |
| Partición Beijing (paper) | train = 2010–2013 (4 años), test = 2014 completo | sección "Data description" |
| Resultado Beijing t+1 (referencia) | RMSE=19.04, MAE=11.13, R²=0.937 | Tabla 1 |
| Framework (texto del paper) | El texto es ambiguo/parcialmente contradictorio: "We adopt ... Pytorch ... and Sklearn ... to build **all** machine learning methods" y, en la misma sección, "the open-source Tensorflow library is used to configure deep learning and Transformer models" — ver nota debajo, no se resuelve esta ambigüedad, se documenta tal cual | Implementation details |
| Hardware (paper) | NVIDIA Quadro P6000, 24GB | Implementation details |

## B. Parámetros inferidos del texto o diagrama (no tabulados explícitamente)

| Parámetro | Lectura | Grado de certeza |
|---|---|---|
| Estructura de doble rama del encoder | Rama 1: un solo bloque de sparse attention; Rama 2: dos bloques de sparse attention en cascada con Conv1D+max-pool (destilado). La Rama 1 "recibe una entrada reducida a la mitad, 1/2 Lx". Las dos ramas se concatenan al final. | **Media-baja.** El texto describe el concepto pero no especifica cómo se concatenan exactamente las dos ramas (ni sobre qué eje) ni por qué la Rama 1 usa media longitud. Se interpreta como una variante del "distilling" de Informer (múltiples réplicas del encoder sobre longitudes decrecientes), pero la mecánica exacta requiere una decisión de implementación nuestra — documentada como tal en el código cuando se escriba. |
| Selección Top-u de queries | "solo las u queries dominantes se usan para calcular self-attention, rellenando los demás pares con cero" | **Media.** Confirma el mecanismo (ProbSparse de Informer), pero no da la fórmula de u ni su valor — ver categoría C. |
| Uso de dropout en STN | El paper especifica dropout solo para los modelos de comparación (CNN: 0.3; BiLSTM: 0.05), nunca para STN/Transformer | **Alta** confianza en que *no* está reportado para STN — se lista en C. |

## C. Parámetros NO especificados por el paper — decisión nuestra

Estos valores **no aparecen en ninguna parte del artículo** (ni texto, ni tablas, ni figuras).
Deben decidirse y documentarse aquí antes de escribir `src/models/stn.py`, y nunca
presentarse como "según el paper".

| Parámetro | Estado | Nota |
|---|---|---|
| `d_model` (dimensión de embedding) | No especificado | A decidir; candidato inicial pequeño (p.ej. 32–64) dado el input de 3-8 variables y el requisito de "tamaño moderado" de la guía del curso |
| Número de cabezas de atención | No especificado | A decidir |
| `d_ff` (dimensión interna feed-forward) | No especificado | A decidir |
| Dropout (STN) | No especificado para STN (solo para baselines CNN/BiLSTM) | A decidir; los valores de los baselines (0.3, 0.05) pueden usarse como referencia de rango, no como valor reportado |
| Factor/fórmula de Top-u | No especificado — ni el valor de `u` ni la constante de muestreo | Resuelto tomando prestada la convención de Informer (`u = factor · ceil(ln(L))`, ver sección "SparseAttention implementation decision" más abajo para la fórmula exacta con su doble `ceil` y los valores de `factor` aún sin aprobar para los experimentos) |
| Número de claves muestreadas (L̃) | No especificado | A decidir |
| Longitud de entrada del decoder / número de start tokens | No especificado | A decidir |
| Criterio exacto de outliers | Solo se menciona "eliminating outliers" en la Fig. 2 (metodología general), sin umbral ni método | A decidir (o justificar por qué no se aplica, dado que la Primera Entrega ya no imputa/elimina outliers de forma agresiva) |
| Inicialización de pesos | No especificado | A decidir (Keras default = Glorot uniform, salvo que se justifique otra cosa) |
| Semilla aleatoria | No especificado | El paper no reporta semilla; nosotros usaremos {42, 123, 2026} — ver docs/PHASE2_PLAN.md |
| Early stopping: métrica, patience, min_delta | No especificado — el paper solo dice "máximo de 200 épocas", sin mencionar si usa early stopping | A decidir |
| Weight decay / regularización adicional | No mencionado | A decidir si se necesita |

## Nota sobre normalización — alcance del ajuste (min/max)

El paper dice únicamente "all air quality time series data are normalized to [0, 1]"
(Implementation details). **No especifica** sobre qué partición se calculan los mínimos y
máximos usados para esa normalización — no dice si se ajustan solo con train o con la serie
completa (train+val+test). No debe afirmarse que el paper ajustó sus parámetros de
normalización con toda la serie; eso no está en el texto.

**Nuestra implementación:** siguiendo la misma regla anti-fuga ya aplicada al `StandardScaler`
de la línea base Ridge (`src/data/fontibon.py`, verificado en `tests/test_scaler_leakage.py`),
el `MinMaxScaler` de la STN se ajustará **exclusivamente sobre train** y se aplicará después,
sin reajustar, a validation y test. Esto es una decisión nuestra por higiene metodológica, no
una réplica de un procedimiento que el paper describa.

## Nota sobre el framework (PyTorch/Sklearn/TensorFlow)

El texto de Implementation Details del paper es internamente ambiguo, casi contradictorio: en
la misma sección afirma que adoptan "Pytorch ... and Sklearn ... to build all machine learning
methods for air quality forecasting" (declaración general, "all"), e inmediatamente después
that "the open-source Tensorflow library ... is used to configure deep learning and Transformer
models" (declaración específica para los modelos profundos, incluyendo STN/Transformer). No
intentamos resolver esta ambigüedad a favor de uno u otro framework — se documenta tal como
aparece en el texto.

Esto es irrelevante para nuestra implementación: la guía del curso (Sección 3) exige
TensorFlow/Keras, y esa es la que usaremos, independientemente de cuál haya usado el paper.
No se está "replicando" el framework en ningún caso, únicamente la arquitectura y los
hiperparámetros reportados.

## SparseAttention implementation decision

Implementada en `src/models/sparse_attention.py`, probada en 47 tests en
`tests/test_sparse_attention.py`. Los valores usados en esos tests (d_model=32, heads=4,
factor=3) son **valores de prueba de software**, no hiperparámetros aprobados para los
experimentos — ver `docs/PHASE2_PLAN.md` §7. Revisión de esta sección: segunda ronda
(non_selected_fill default cambiado a zero, fórmula sample_k/n_top corregida, muestreo por
query, RNG con estado, causal_mode="probsparse"/"strict" — ver más abajo por qué).

### ZHANG & ZHANG (2023) — qué reporta el paper, literalmente

- Ec. 7 (aproximación por muestreo de la medida de sparsity):
  `M̃(q_i, K̃) = max_j(q_i·k_j / √d) − mean_j(q_i·k_j / √d)`, sobre un subconjunto muestreado
  `K̃` de keys.
- Selección de las `u` queries dominantes según ese puntaje.
- Las posiciones no seleccionadas se describen textualmente como rellenadas **con cero**
  ("filling other pairs with zero").
- El decoder usa un bloque de **masked sparse attention** — es decir, la misma mecánica de
  atención dispersa, pero con máscara causal.
- **NO define** ninguna fórmula ni valor para `u` (número de queries top) ni para el tamaño
  del muestreo de keys.

### INFORMER (Zhou et al. 2021) — qué resuelve la ambigüedad, citado por el propio paper

- Resuelve `u`/`sample_k` mediante `factor · ceil(ln(L))` (su constante `c`, aquí expuesta
  como `factor`).
- Muestrea las keys **por query** (un subconjunto distinto de keys para cada fila de `Q`, no
  un único subconjunto compartido entre todas las queries) — compartido entre batch y heads
  por eficiencia, pero no entre posiciones de query.
- Usa un contexto alternativo (no cero) para las queries no seleccionadas: `mean(V)` en el
  encoder (no causal), `cumsum(V)` (suma acumulada, sin dividir) en su atención enmascarada de
  decoder.
- Su propio muestreo/selección top-u en modo enmascarado **no está restringido causalmente**:
  solo el cómputo final de atención de las queries seleccionadas lleva máscara causal.

### NUESTRA RÉPLICA R1 — lo que implementa `SparseAttention` por defecto (`causal_mode="probsparse"`)

- **Zero como comportamiento principal** (`non_selected_fill="zero"` es el default): prioriza
  fidelidad textual al paper para la réplica R1, por encima de la conveniencia de gradiente que
  ofrecía `mean(V)`. No se afirma que zero-fill elimine necesariamente el gradiente de esas
  posiciones en el bloque STN completo: el paper especifica conexiones residuales alrededor de
  cada sub-capa ("a residual connection is used around each of two sub-layers"), que en el
  bloque completo (no implementado aún) le darían a cada posición una vía de gradiente
  independiente de esta capa. Esta capa, aislada, no tiene esa conexión residual todavía.
- **Fórmula de Informer para resolver `u`/`sample_k`**: `compute_sample_k`/`compute_n_top`
  calculan `ceil(factor · ceil(ln(L)))`, clipeado a `[1, L]`. La doble aplicación de `ceil`
  (primero sobre `ln(L)` para obtener un entero base, luego sobre `factor · base` para
  garantizar un entero final incluso con `factor` flotante) es una convención nuestra explícita
  para que la fórmula quede bien definida cuando `factor` no es entero — verificada con valores
  conocidos: `L=24, factor=5 → 20`; `L=24, factor=3 → 12`
  (`tests/test_sparse_attention.py::test_sample_k_and_n_top_exact_known_values`).
- **Muestreo por query**: `_sample_key_indices_per_query` devuelve un tensor `(L_query,
  sample_k)`, no `(sample_k,)` — cada query tiene su propio subconjunto aleatorio de keys,
  compartido entre batch y heads (no entre posiciones), replicando la práctica real de
  Informer.
- **Selección top-u global** (no restringida por prefijo causal) incluso cuando
  `causal=True`: la competencia por el top-u ocurre sobre las `L_q` queries completas, igual
  que en el paper/Informer. La máscara causal se aplica únicamente al cómputo final de
  atención de las queries que resultan seleccionadas (sus logits contra keys futuras quedan en
  `-inf` antes del softmax) — verificado en
  `test_probsparse_causal_attention_weights_never_attend_future_keys`.
  - **Consecuencia documentada, no un bug**: como en el propio Informer, si una query `i` es o
    no seleccionada como "top" puede verse influido por los puntajes de queries futuras
    `j > i`, porque el muestreo/ranking no está restringido causalmente. Esto es una
    característica interna del algoritmo de selección, **no una fuga de datos objetivo**: el
    valor de atención real que recibe cualquier query nunca pondera una key futura (eso sí está
    garantizado y probado). La ausencia de fuga real (target/datos futuros) se garantiza en
    otra capa del diseño, ya prevista pero aún no implementada: el input del decoder nunca
    contendrá PM2.5 futuro real, los tokens de pronóstico futuros serán placeholders/ceros, y
    no se usará teacher forcing con valores futuros desconocidos. Un test dedicado
    (`test_probsparse_causal_selection_can_depend_on_future_scores_known_limitation`) deja
    constancia de que este comportamiento existe y es esperado, sin tratarlo como fallo.
- **Masked sparse attention en el decoder**: la misma mecánica anterior, con
  `causal=True` — sigue siendo dispersa (top-u), no atención densa, replicando la arquitectura
  descrita por Zhang & Zhang para el decoder.

### VARIANTES EXPERIMENTALES (no son la réplica principal de R1)

- **`non_selected_fill="mean"`**: usa `mean(V)` (no causal) o una media acumulada causal de
  `V` (causal — normalizada, a diferencia del `cumsum` sin dividir de Informer, para mantener
  la magnitud comparable al `mean(V)` no causal en vez de crecer sin límite con la posición).
  Disponible como opción configurable, nunca por defecto.
- **`causal_mode="strict"`**: variante experimental que además restringe el muestreo de keys y
  la competencia top-u al prefijo causal propio de cada query (`0..i`), eliminando incluso la
  dependencia algorítmica de la selección frente a datos futuros descrita arriba — a costo de
  computar atención causal densa completa (`O(L²)`, aceptable dado que `L` es pequeño en este
  proyecto: ≤24 en el encoder, y el decoder será aún más corto por ser horizonte t+1). Se
  conserva para comparación (verificado en `test_strict_causal_mode_output_independent_of_future_input`,
  que exige exactamente la propiedad que `"probsparse"` no garantiza), pero **no** es la
  variante que se usa para la réplica R1.

### Reproducibilidad — diseño de RNG, entrenamiento estocástico / evaluación determinista

Una primera versión usaba un par de semilla determinista fijo por instancia (`[self.seed, 0]`),
lo que hacía que una misma capa muestreara exactamente los mismos índices en **toda** llamada,
tanto en entrenamiento como en evaluación. Se corrigió en dos rondas:

1. Se introdujo `tf.random.Generator.from_seed(seed)`: un generador con estado propio por
   instancia que avanza en cada llamada, dando muestras distintas en llamadas sucesivas.
2. Pero eso rompía la evaluación: si `validation`/`test` llaman a la capa repetidamente (o el
   pipeline de evaluación se ejecuta más de una vez), el muestreo interno de
   `SparseAttention` introducía ruido no deseado en las métricas. Se separó explícitamente el
   comportamiento según `training`:
   - **`training=True`** → muestreo **estocástico**, vía `self._rng` (el generador con estado
     de la instancia); dos instancias construidas con la misma semilla, mismos pesos y la
     misma secuencia de llamadas `training=True` reproducen la misma secuencia de salidas
     (`test_same_seed_same_call_sequence_reproduces_across_instances_during_training`), pero
     una misma instancia puede muestrear keys distintas entre llamadas sucesivas
     (`test_sampling_can_differ_between_successive_training_calls_on_same_instance`).
   - **`training=False` o `training=None`** → muestreo **determinista**, vía
     `tf.random.stateless_uniform` con un par de semilla fijo por instancia
     (`self._eval_seed_pair`, derivado de `seed` o de un valor de respaldo si `seed=None`) que
     nunca avanza: la misma entrada, mismos pesos, misma instancia, llamada tantas veces como
     se quiera, produce exactamente la misma salida
     (`test_evaluation_sampling_is_deterministic_across_repeated_calls`). `training=None` se
     trata igual que `training=False` (`test_training_none_matches_training_false`) — evaluación
     por defecto, nunca estocástico salvo pedido explícito.

### Proyección de salida sin bias (zero-fill)

Con `non_selected_fill="zero"`, la mayoría de posiciones de query (todas menos las `n_top`
seleccionadas) llegan a la proyección final `Dense(d_model)` con un contexto exactamente cero.
Un bias aprendido en esa capa convertiría ese cero documentado en un valor no-cero para cada una
de esas posiciones, rompiendo la semántica de "zero-fill" descrita en el paper. Se configuró
`self.dense = tf.keras.layers.Dense(d_model, use_bias=False, ...)` — únicamente la proyección de
salida; `Wq`/`Wk`/`Wv` conservan su bias por defecto, no hay razón para restringirlos. Verificado
en `test_output_projection_has_no_bias` y, de punta a punta, en
`test_zero_fill_positions_produce_exact_zero_output` (una posición no seleccionada en ningún
head produce un vector de salida final exactamente cero).

### Shapes estáticas

`SparseAttention` exige que la dimensión de longitud de secuencia sea estática (conocida en
tiempo de construcción del grafo/llamada) — no soporta longitudes dinámicas. La comprobación de
`None` ocurre **antes** de convertir la forma a `int`. Esto es aceptable para el proyecto: todas
las ventanas usadas (Beijing, Fontibón) tienen longitud fija.

### Pendiente

- `d_model`, `num_heads`, `factor` definitivos para los experimentos siguen sin decidirse — ver
  CANDIDATE_V1 más abajo, que fija valores de prueba para poder construir y ejecutar la STN,
  explícitamente no aprobados como definitivos.

## STN architecture implementation decisions

Implementada en `src/models/stn.py`, probada en 32 tests en `tests/test_stn.py` + smoke test
en `scripts/smoke_test_stn.py` (99 425 parámetros entrenables con CANDIDATE_V1, forward/loss/
gradientes/optimizer-step verificados sobre datos sintéticos — nunca Beijing ni Fontibón). Solo
arquitectura: **no se ha entrenado nada real todavía**.

### REPORTADO POR PAPER (Zhang & Zhang, 2023)

- Mecanismo de sparse attention (ver sección "SparseAttention implementation decision" arriba).
- Estructura encoder-decoder.
- Positional embedding sinusoidal.
- Conv1D con kernel=3 en la rama de destilado del encoder.
- Max-pooling con stride=2.
- Profundidad original: 3 encoders / 2 decoders.
- El decoder usa **masked sparse self-attention** (la misma mecánica de `SparseAttention`, con
  `causal=True`).
- El decoder usa una capa de **atención cruzada multi-head estándar** (no dispersa) — implementada
  literalmente con `tf.keras.layers.MultiHeadAttention`, no con `SparseAttention`.
- Conexiones residuales + normalización alrededor de cada sub-capa.
- Mapeo lineal final para la salida.

### PROMETIDO EN FASE 1 (Primera Entrega)

- Versión **reducida** de la STN (profundidad de encoder-decoder disminuida), anunciada
  explícitamente en `Primer Entrega DL.pdf` antes de implementar nada.
- Horizonte t+1 (predicción de una sola hora futura), no los horizontes de 6/12/24/48h que sí
  evalúa el paper.

### CANDIDATE V1 — decisiones nuestras, NO definitivas (`configs/stn_candidate_v1.yaml`)

| Decisión | Procedencia |
|---|---|
| `n_encoder_units=2`, `n_decoder_layers=1` (en vez de 3/2) | Desviación deliberada frente al paper, ya anunciada desde la Primera Entrega — la clase `STNModel` sigue soportando 3/2 vía sus parámetros `n_encoder_units`/`n_decoder_layers` si se quisiera aproximar la profundidad original (`test_encoder_paper_compatible_three_units_shape`) |
| `d_model=32` | Decisión nuestra: modelo moderado, sin búsqueda de hiperparámetros |
| `num_heads=4` (head_dim=8) | Decisión nuestra |
| `d_ff=128` | Decisión nuestra: relación 4×d_model, convención típica de Transformer |
| `dropout=0.10` | Decisión nuestra: el paper no reporta dropout para STN |
| `factor=3` | Decisión nuestra, inspirada en ProbSparse/Informer — NO reportada por Zhang & Zhang |
| `decoder_label_len=12` | Decisión nuestra: aproximadamente la mitad de la ventana histórica de 24h como tokens de inicio (estilo Informer); el paper no da la longitud exacta |
| Activación ReLU en el feed-forward | Decisión nuestra, convención estándar de Transformer; el paper no especifica la activación |
| `padding="same"` en el Conv1D de destilado | Decisión nuestra (operativa, no reportada) |
| Interpretación de la rama corta: últimos `L/2` pasos temporales | Nuestra lectura de la Fig. 3, no un hecho explícito del paper — ver más abajo |
| Concatenación de ramas sobre el eje temporal | Nuestra lectura de la Fig. 3, no un hecho explícito del paper — ver más abajo |

Ningún valor de esta tabla se presenta en el código, los configs o el informe como "reportado
por el paper".

### Interpretación de las dos ramas del encoder (Fig. 3)

El paper describe conceptualmente un encoder con dos "pipelines paralelos" (uno con un solo
bloque de sparse attention, otro con dos bloques de sparse attention en cascada con Conv1D +
max-pooling), y dice que la primera rama "recibe una entrada reducida a la mitad" — pero **no
especifica inequívocamente** qué mitad exacta recibe la rama corta ni sobre qué eje se
concatenan las dos ramas al final. Nuestra interpretación, implementada en `EncoderUnit`
(`src/models/stn.py`):

- **Rama larga**: las 24 posiciones completas → `SparseAttentionBlock` → `Conv1D(kernel=3,
  same)` → `MaxPool1D(pool=2, stride=2)` → `SparseAttentionBlock` → termina en 12 posiciones.
- **Rama corta**: las **últimas** 12 posiciones temporales de la entrada cruda de la unidad
  (`x[:, -12:, :]`) → `SparseAttentionBlock` → 12 posiciones. Se tomaron las últimas (no las
  primeras) porque son las temporalmente más próximas a la hora objetivo, y por tanto las más
  informativas para el pronóstico de t+1.
- **Concatenación**: sobre el eje **temporal**, rama larga primero y luego rama corta
  (`tf.concat([long, short], axis=1)`), recomponiendo una longitud de 24. Se concatena en el
  eje temporal (no en el de features) porque ambas ramas ya comparten `d_model` tras sus
  respectivos bloques de atención, y esto preserva la interpretación de "24 posiciones
  temporales" que espera el resto de la arquitectura (positional encoding, decoder
  cross-attention).

Verificado en `test_encoder_unit_branch_lengths` (24→12/12, concat→24) y
`test_encoder_unit_shape` ((B,24,d_model)→(B,24,d_model)).

### Decoder input (diseño anti-fuga)

Para t+1 con `lookback=24`, `decoder_label_len=12`, `prediction_length=1`: el input del decoder
se construye exclusivamente a partir de las **últimas 12 horas observadas** de PM2.5
normalizado (posiciones t-11..t) más **1 placeholder = 0** para la posición t+1, dando forma
`(B, 13, 1)`. Implementado en `build_decoder_input` (`src/models/stn.py`), que **no tiene ningún
parámetro para el target futuro real**: no puede filtrarlo porque nunca se le entrega. Verificado
explícitamente en `test_decoder_input_never_uses_future_target` (construye un historial donde
cada posición vale su propio índice temporal, confirma que el resultado es exactamente
`[12..23, 0]`, que un valor centinela "futuro" nunca aparece, y que la firma de la función no
acepta ningún argumento `y`/`target`/`future`).

Esta es la pieza de diseño que hace aceptable, para este proyecto, que la selección top-u de
`causal_mode="probsparse"` no esté restringida causalmente en su ranking interno (ver sección
"SparseAttention implementation decision" arriba): la ausencia real de fuga de datos objetivo
no depende de esa restricción algorítmica, sino de que el decoder jamás recibe el valor real de
PM2.5 en t+1 en primer lugar.

### Nombres estructurales para transferencia futura

`STNModel` expone `model.encoder`, `model.decoder`, `model.output_head` como atributos con
nombres estables (`encoder_input_projection`, `encoder_unit_0`, `encoder_unit_1`,
`decoder_layer_0`, ...). Esto es deliberado: permitirá que los experimentos A/B/C/D (todavía no
implementados — ver `docs/PHASE2_PLAN.md` §7) congelen/descongelen/reinicialicen submódulos
completos (p. ej. `model.encoder.trainable = False`, o reemplazar `model.output_head` por una
cabeza nueva para Config B) sin necesitar hacks de introspección de la arquitectura.

### Semillas únicas por bloque de SparseAttention

Antes de esta ronda, la misma `seed` base se propagaba sin cambios a **todas** las instancias
de `SparseAttention` dentro del encoder y el decoder. Dos bloques distintos con la misma
longitud temporal (p. ej. dos `SparseAttentionBlock` dentro de la misma `EncoderUnit`) podían
así generar secuencias de muestreo idénticas, algo no deseado. Se corrigió con
`derive_child_seed(base_seed, offset)` (`src/models/stn.py`): una función determinista y
estable entre procesos (nunca `hash()` de Python, que varía por ejecución) que combina la
semilla base con un offset entero fijo por bloque. Esquema de offsets documentado en el código:
`encoder_unit[i]` → `long_block_1 = i·10+1`, `long_block_2 = i·10+2`, `short_block = i·10+3`;
`decoder_layer[i]` → `self_attn = 1000 + i·10 + 1`, `cross_attn = 1000 + i·10 + 2`. `base_seed=None`
siempre produce `child_seed=None` (cada bloque cae de vuelta al generador global de TensorFlow,
igual que antes). Verificado en `tests/test_stn.py`: bloques distintos tienen semillas
distintas; reconstruir el mismo modelo con la misma semilla base reproduce exactamente el mismo
mapa de semillas hijas; cambiar la semilla base cambia todas las semillas hijas; `training=False`
sigue siendo determinista con este esquema; el número de parámetros no cambia.

### `MultiHeadAttention` (cross-attention del decoder) necesitó semilla explícita

Al implementar el test `dos modelos construidos después de resetear la misma seed → mismos
pesos iniciales` (`tests/test_trainer.py`) se detectó que, tras entrenar un primer modelo en el
mismo proceso (lo que construye y usa su capa `tf.keras.layers.MultiHeadAttention` de
cross-attention), un `tf.keras.utils.set_random_seed(...)` posterior ya **no** reproducía los
mismos pesos iniciales para esa capa específica en un segundo modelo — mientras que `Dense`,
`Conv1D` y las proyecciones internas de `SparseAttention` sí se reproducían correctamente. Es
decir: `MultiHeadAttention` no se reinicializa de forma confiable solo con el reseteo de semilla
global después de que otra instancia ya fue construida y usada antes en el proceso. Se corrigió
pasando un `kernel_initializer=tf.keras.initializers.GlorotUniform(seed=...)` con una semilla
explícita derivada vía `derive_child_seed` — consistente con cómo ya se siembra el resto del
modelo, y ya no depende del estado aleatorio global de TensorFlow en absoluto. Verificado: el
test que originalmente falló ahora pasa (0/78 variables difieren, antes 4/78).

### Pendiente

- Ningún experimento A/B/C/D implementado todavía.
- Ningún entrenamiento real de 200 épocas ejecutado todavía — solo un pilot corto de integración
  sobre Beijing train/validation (ver sección "Beijing preprocessing (R1)" y
  "Pilot R1 — no es resultado final" más abajo).
- Fontibón no se ha usado en esta etapa.
- `d_model`, `num_heads`, `d_ff`, `dropout`, `factor`, `decoder_label_len` definitivos siguen
  sin decidirse — CANDIDATE_V1 es solo para validar que la arquitectura construye y entrena, no
  una elección final.

## Beijing preprocessing (R1)

Implementado en `src/data/beijing.py: prepare_beijing_r1`, probado en `tests/test_beijing_pipeline.py`
(24 tests). Reutiliza deliberadamente la lógica ya auditada de Fontibón
(`src/data/fontibon.py: interpolate_short_gaps`, `build_windows`, generalizada con un parámetro
`target_col` para no romper el comportamiento por defecto de Fontibón) en vez de reimplementarla,
porque la regla anti-fuga que necesitamos es idéntica: partición cronológica antes de imputar,
el objetivo nunca se imputa, solo se interpolan huecos internos ≤6h en las variables de entrada,
se registra el ancla derecha de cada interpolación, y ninguna ventana puede usar información en o
después de la hora objetivo.

### Eliminación de outliers — deliberadamente NO implementada

El paper menciona "eliminating outliers" en su diagrama de metodología general (Fig. 2), pero no
especifica ningún criterio, umbral ni método. Siguiendo la misma regla que ya aplicamos en
`docs/IMPLEMENTATION_ASSUMPTIONS.md` categoría C ("nunca inventar un valor no reportado y
presentarlo como si lo fuera"), **no se implementa ninguna eliminación de outliers para R1**.
En particular: no se recorta (`clip`) el PM2.5 alto — el máximo observado (994 µg/m³, ver
`docs/PHASE1_AUDIT.md` §6) puede representar un episodio real de contaminación extrema en
Beijing, no necesariamente un error de sensor, y recortarlo arbitrariamente introduciría un sesgo
que el paper no describe ni justifica cómo evitar.

### Protocolo de faltantes — adaptación deliberada frente al paper

El paper dice que hace "interpolación de faltantes" pero **no describe en absoluto** cómo evita
la fuga de información durante esa interpolación (p. ej. si usa datos de validación/test para
interpolar train, o si permite que una interpolación use el propio valor objetivo). Nuestro
protocolo, inspirado en el pipeline ya auditado de Fontibón:

1. Particionar train/validation/test **antes** de cualquier imputación.
2. Nunca usar observaciones de otra partición para imputar una partición distinta (garantizado
   por construcción: la interpolación ocurre dentro de `X.iloc[inicio:fin]`, ya recortado a una
   partición).
3. El PM2.5 usado como **objetivo** nunca se imputa.
4. Solo se interpolan huecos internos de **hasta 6 horas** en las variables de entrada (De las 8
   variables Beijing, solo PM2.5 presenta valores faltantes — 2067, 4.72% — las otras 6
   variables continuas y `cbwd` no tienen ninguno; ver `docs/PHASE1_AUDIT.md` §6, por lo que en
   la práctica la interpolación solo actúa sobre PM2.5-como-entrada).
5. Se registra el ancla derecha usada en cada interpolación.
6. Una ventana es válida solo si ninguna interpolación de su historial usó información en la
   hora objetivo o posterior.
7. Los huecos mayores a 6h permanecen como faltantes y provocan el descarte de toda ventana que
   los toque.
8. Ninguna ventana cruza fronteras entre particiones (garantizado por construcción, igual que en
   Fontibón).

Esta es una **desviación deliberada frente al paper**, no una réplica de un procedimiento que el
paper describa — porque el paper no describe ninguno con el detalle suficiente para evitar fuga.

### Codificación categórica y escalado — ajustados solo con train

`cbwd` se codifica con `sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")`, **ajustado
únicamente con las filas de train**. En la práctica las 4 categorías (NE, NW, SE, cv) aparecen
todas en train, así que `handle_unknown="ignore"` nunca se activa con los datos reales, pero
queda como salvaguarda. El escalado usa dos `MinMaxScaler` **ajustados solo con train**: uno
exclusivo para PM2.5 (`pm25_scaler`) y otro para las 6 variables continuas restantes
(`other_scaler`) — separados deliberadamente para poder invertir la transformación de PM2.5 sin
tocar ni necesitar las demás variables (`BeijingScalers.inverse_transform_pm25`, verificado con
un test de roundtrip exacto). Las columnas one-hot permanecen 0/1, sin volver a escalarlas.

### Pilot R1 — no es resultado final

`scripts/run_r1_pilot.py` ejecuta como máximo 5 épocas sobre un subconjunto cronológico (no
aleatorio) de hasta 4096 ventanas de train y 1024 de validación, únicamente para verificar que
la cadena completa (CSV → preprocesamiento → dataset → STN → entrenamiento → validación →
inverse transform → métricas → checkpoint → TensorBoard → normas de gradiente) funciona de
punta a punta. **No mide desempeño final** y así se rotula explícitamente en su salida y en el
notebook ("PILOT DE INTEGRACIÓN — NO RESULTADO FINAL"). Beijing test 2014 se prepara únicamente
para contar sus ventanas en la auditoría de preprocesamiento
(`outputs/metrics/beijing_preprocessing_audit.json`); nunca se pasa al modelo.

## Trainer — endurecimiento antes del entrenamiento completo de R1

### `optimization_loss_mean` vs. `train_loss`/`val_loss` deterministas

Versión previa del trainer: `train_loss` se calculaba como la media de las pérdidas MSE
observadas durante los batches de optimización de la época (`training=True`, pesos cambiando
batch a batch), mientras que `val_loss` se calculaba **después**, con los pesos finales de la
época y `training=False`. Estos dos números no eran comparables — uno mide un objetivo móvil
(los pesos intermedios), el otro los pesos finales. Se corrigió separando explícitamente tres
cantidades por época (`src/training/trainer.py`):

- **`optimization_loss_mean`**: media de las pérdidas de los batches durante la optimización
  (`training=True`, muestreo estocástico de `SparseAttention`). Diagnóstico adicional únicamente
  — nunca se compara directamente contra `val_loss`.
- **`train_loss`**: MSE determinista (`training=False`) calculado **después** de terminar la
  época, con los pesos finales, sobre **todo** TRAIN.
- **`val_loss`**: igual, sobre todo VALIDATION.

`train_rmse_scaled = sqrt(train_loss)` y `val_rmse_scaled = sqrt(val_loss)` — por construcción,
así que `train_rmse_scaled² ≈ train_loss` y `val_rmse_scaled² ≈ val_loss` se cumplen siempre
dentro de tolerancia numérica (verificado en
`tests/test_trainer.py::test_train_rmse_scaled_squared_equals_train_loss`). Las curvas
principales de la entrega son `train_loss` vs `val_loss` y `train_rmse_scaled` vs
`val_rmse_scaled` — **no** `optimization_loss_mean`, que queda solo como diagnóstico.

### Evaluación por batches (`evaluate_split_batched`)

Antes: una sola llamada `model((split.encoder_input, split.decoder_input))` sobre la partición
completa. Con las 29 237 ventanas de train de Beijing R1 (y eventualmente más para
Fontibón/R2), esto arriesga problemas de memoria. Se corrigió con
`evaluate_split_batched(model, split, batch_size)` (`src/training/trainer.py`): usa
`tf.data.Dataset` con `shuffle=False` y el mismo `batch_size` del entrenamiento, devolviendo las
predicciones en el mismo orden que la partición original. Verificado
(`test_batched_predictions_match_direct_predictions`) que, con `training=False`, la predicción
por lotes coincide exactamente con la predicción directa sobre un dataset sintético pequeño.

### Semilla ANTES de construir el modelo

`Trainer.fit()` llamaba `set_random_seed(seed)` en su primera línea — para ese momento el modelo
ya podía estar construido (pesos ya inicializados), así que esa llamada no podía hacer
reproducible la inicialización de pesos, solo el shuffling posterior de los datos. Se creó
`src.utils.seeding.set_experiment_seed(seed)`, que debe ejecutarse **antes** de construir
`STNModel`, con el orden documentado: `set_experiment_seed(seed) → construir modelo → construir
Trainer → fit`. `Trainer` sigue llamando a `set_experiment_seed` internamente al inicio de
`fit()` como segunda línea de defensa (para el shuffling), pero se documenta explícitamente que
no sirve para la inicialización de pesos si el modelo ya existía.

Al implementar el test correspondiente se encontró un problema real, no solo de test: la capa
`tf.keras.layers.MultiHeadAttention` (cross-attention del decoder) no se reinicializaba de forma
confiable solo con el reseteo de semilla global, si ya se había construido y usado otra instancia
antes en el mismo proceso — ver la sección "STN architecture implementation decisions" más
arriba, "`MultiHeadAttention` necesitó semilla explícita", para la causa raíz y la corrección
(`kernel_initializer` con semilla derivada explícita).

### Persistence baseline como sanity check (no como línea base de la Primera Entrega)

`persistence_prediction(split, feature_names)` (`src/data/beijing.py`) predice
PM2.5(t+1) = PM2.5(t) — el último valor ya presente en el historial de cada ventana, sin
recomputar nada. Se calcula únicamente sobre Beijing **validation**, nunca sobre test 2014, y se
guarda en `outputs/metrics/R1_beijing_validation_baselines.json` con una aclaración explícita:
esto **no** es la línea base seleccionada en la Primera Entrega (esa fue Ridge, sobre Fontibón) —
es exclusivamente un sanity check para R1 sobre Beijing, cuyos niveles y dinámica son muy
distintos a los de Fontibón.

## R1 — evaluación final sobre Beijing test 2014 (protocolo congelado)

Después de que las tres semillas de R1 development (42, 123, 2026) mostraran estabilidad
suficiente en validación (CV del RMSE ≈1.1%), el protocolo completo se congeló **por escrito,
antes de tocar test** — `outputs/metrics/R1_frozen_protocol.json`, generado por
`scripts/run_r1_test_evaluation.py` antes de cualquier lectura de Beijing test 2014. Ese archivo
documenta arquitectura, hiperparámetros de entrenamiento, semillas y rutas de checkpoint
exactas, de modo que la evaluación de test quede trazable a un protocolo fijado de antemano, no
reconstruido después de ver el resultado.

La evaluación de test en sí (`scripts/run_r1_test_evaluation.py`) es, para cada semilla,
estrictamente: reconstruir `STNModel` con la configuración congelada → cargar
`best.weights.h5` → `evaluate_split_batched(..., training=False)` sobre las 8 487 ventanas de
test → `inverse_transform` de PM2.5 → métricas en µg/m³. **No hay optimizador, no hay
`GradientTape`, no hay ninguna llamada con `training=True` en todo el script.** Se ejecutó una
sola vez; no se relanzó tras ver el resultado.

**Comparación con Zhang & Zhang (2023):** se reporta como diferencia descriptiva
(`nuestro_valor − valor_publicado`), nunca como evidencia de que la arquitectura es mejor o
peor — las diferencias metodológicas ya documentadas a lo largo de este archivo (STN reducida
2/1 vs 3/2, validación cronológica vs aleatoria, protocolo anti-fuga propio para faltantes,
MinMax ajustado solo con train frente a un alcance no especificado en el paper, y los valores
propios de `d_model`/`heads`/`d_ff`/`dropout`/`factor`/`decoder_label_len`) hacen que esto sea
un resultado de referencia publicado, no un benchmark perfectamente equivalente.

## R2 — preentrenamiento source para transferencia (Beijing → Fontibón)

**Propósito de R2, distinto del de R1:** R1 fue la réplica/adaptación del paper sobre las 8
variables originales de Beijing, cerrada y evaluada una sola vez contra Beijing test 2014 (ver
sección anterior). R2 **no es una réplica ni busca comparar contra el paper**: su único
propósito es producir un `STNModel` preentrenado sobre Beijing usando **exclusivamente las 3
variables que Beijing y Fontibón comparten** (PM2.5, temperatura, punto de rocío), para servir
como fuente de pesos en los futuros experimentos de transferencia Config B/C/D (todavía no
implementados — ver `docs/PHASE2_PLAN.md` §5, §7). R2 nunca toca Beijing test 2014 ni Fontibón
(validación o test) — su propio proceso de desarrollo usa el mismo split train/validation
cronológico ya aprobado para R1 (ver `docs/PHASE2_PLAN.md` §2).

### Por qué los checkpoints de R1 no se reutilizan como punto de partida de R2

R1 se entrenó con **11 canales de entrada** (8 variables continuas + one-hot de 4 categorías de
`cbwd`, ver `src/data/beijing.py:prepare_beijing_r1`), mientras que R2 usa **3 canales**
(`pm25`, `temp`, `dew`). La primera capa de la STN es una proyección de entrada
(`Dense`/proyección lineal) cuya matriz de pesos tiene forma `(n_features_entrada, d_model)` —
`(11, 32)` en R1 frente a `(3, 32)` en R2 con CANDIDATE_V1. Estas matrices **no son compatibles
por forma**: no existe una manera de recortar o rellenar `(11, 32)` en `(3, 32)` que preserve un
significado entrenado, porque las columnas de la matriz de R1 no corresponden 1 a 1 con las 3
variables de R2 (el one-hot de `cbwd` no tiene equivalente en absoluto). Por esta razón, **R2
entrena su proyección de entrada — y todo el resto del modelo — desde cero**, con su propia
semilla (42), nunca cargando `outputs/checkpoints/R1_development_seed*/best.weights.h5`. El
resto de la arquitectura (bloques de atención, decoder, cabeza de salida) es estructuralmente
idéntica entre R1 y R2 porque ambos usan la misma configuración `CANDIDATE_V1` — la
incompatibilidad real está únicamente en la capa de entrada, dependiente del número de
variables.

### Fuente única de verdad para el orden/nombre de las 3 variables compartidas

`src/data/transfer_features.py::TRANSFER_FEATURE_NAMES = ["pm25", "temp", "dew"]` es la única
fuente de verdad, referenciada tanto por `src/data/fontibon.py` (columna final de
`load_fontibon_features`) como por `src/data/beijing.py::prepare_beijing_r2`. Esto fue
necesario porque la auditoría de este round encontró una discrepancia real: `prepare_beijing_r1`
usa internamente el nombre `"dewp"` (no `"dew"`) y lo ordena **antes** de `"temp"`
(`CONTINUOUS_COLS` en `src/data/beijing.py`), mientras que Fontibón ya usaba `"temp"` antes que
`"dew"`. R2 no reutiliza el orden/nombres de R1 precisamente por esto — usa
`TRANSFER_FEATURE_NAMES` de forma independiente, verificado contra el pipeline real de Fontibón
en `tests/test_beijing_r2_pipeline.py::test_beijing_r2_order_matches_fontibon_order` (carga
`load_fontibon_features` real y compara columnas, no es una suposición).

### Escalado — dos `MinMaxScaler` independientes, no un escalador físico compartido

Beijing y Fontibón usan **escaladores separados, cada uno ajustado solo con su propio train**
(`BeijingR2Scalers` en `src/data/beijing.py`, análogo al `FontibonScalers` ya existente). La
compatibilidad entre R2 y Fontibón es **semántica y dimensional** (misma variable física, mismo
orden de canales, mismo rango de normalización [0,1]) — **no** es un escalador físico
compartido: Beijing PM2.5 (media ≈98.6 µg/m³) y Fontibón PM2.5 (media ≈18.3 µg/m³) tienen
distribuciones muy distintas, y ajustar un único escalador sobre ambas mezclaría information de
dominio y sesgaría los rangos. Cada dominio conserva su propio escalador, ajustado solo con su
propio train, siguiendo la misma regla anti-fuga usada en toda la Fase 2.

### Compatibilidad estructural de pesos — auditada, no asumida

`src/models/transfer_compat.py::compare_transfer_weight_shapes(source_model, target_model)`
compara, forma a forma, los pesos de `.encoder` y `.decoder` entre un modelo Beijing-R2 (3
canales) y un modelo con forma Fontibón (3 canales) — ambos construidos sin entrenar, solo para
verificar la forma. `.output_head` se reporta de forma informativa pero **no** cuenta para el
booleano `compatible`, porque en todos los configs de transferencia previstos (B/C/D) la cabeza
de salida se reinicializa siempre para el dominio destino, nunca se copia del source. Ejecutado
en `scripts/run_r2_pilot.py` — resultado: `compatible: true` (96/96 pesos de encoder, 27/27 de
decoder coinciden en forma; ver `outputs/metrics/R2_preprocessing_audit.json`). Un test dedicado
(`test_weight_shape_comparison_detects_real_mismatch`) confirma que el comparador sí detecta una
incompatibilidad real cuando se le da un modelo de 11 canales (forma R1) como destino.

### Pilot de integración R2 — no es resultado final

`scripts/run_r2_pilot.py` ejecutó, una sola vez, un pilot de integración: semilla=42, subconjunto
**cronológico** (no aleatorio) de 4096 ventanas de train y 1024 de validación, máximo 5 épocas.
Su único propósito es verificar que la cadena completa (preprocesamiento de 3 canales → STN
desde cero → entrenamiento → evaluación → inverse transform → métricas en µg/m³) funciona de
punta a punta con la nueva entrada de 3 canales — **no mide desempeño final** y se rotula
explícitamente `"R2 PILOT DE INTEGRACION -- NO RESULTADO FINAL"` en su salida y en
`outputs/metrics/R2_pilot_summary.json`. Beijing test 2014 y Fontibón (validación o test) no se
usaron. No se entrenó con las tres semillas completas (42/123/2026) — eso queda pendiente de
autorización aparte, igual que los experimentos A/B/C/D.

### R2 full source training — las tres semillas (42, 123, 2026)

`scripts/run_r2_source_training.py` entrenó, sin subsampling, las tres semillas aprobadas sobre
el split completo train/validation de R2 (29 237 / 2 841 ventanas), con arquitectura e
hiperparámetros idénticos al pilot (CANDIDATE_V1, Adam, lr=1e-4, batch=32, max_epochs=200,
early stopping monitor=val_loss/patience=15/min_delta=0/restore_best_weights=True). Cada semilla
parte de una inicialización independiente (`set_experiment_seed(seed)` **antes** de construir el
modelo, nunca después) y **nunca reutiliza pesos de otra semilla ni de R1** — ver más arriba,
"Por qué los checkpoints de R1 no se reutilizan como punto de partida de R2".

| Seed | Best epoch | Stopping epoch | Val RMSE (µg/m³) | Val MAE | Val R² | Bias |
|---|---|---|---|---|---|---|
| 42 | 54 | 69 | 21.141 | 13.007 | 0.9473 | +1.634 |
| 123 | 58 | 73 | 20.971 | 12.544 | 0.9481 | +1.329 |
| 2026 | 58 | 73 | 21.068 | 12.646 | 0.9476 | −0.207 |
| **media ± std** | — | — | **21.060 ± 0.085** | **12.732 ± 0.243** | **0.9477 ± 0.0004** | — |

Dispersión entre semillas muy baja (CV del RMSE ≈0.4%). Ninguna semilla se presenta como "la"
mejor — las tres se conservan como checkpoints SOURCE independientes
(`outputs/checkpoints/R2_source_seed{42,123,2026}/best.weights.h5`).

**Interrupción por reinicio del equipo (una vez, sin pérdida de datos):** el proceso se detuvo
por un reinicio de Windows justo cuando seed 42 terminaba y seed 123 empezaba a construirse.
Antes de tocar nada se hizo una auditoría puramente pasiva del estado en disco (sin ejecutar,
borrar ni modificar nada): se confirmó, por el tamaño del archivo de eventos de TensorBoard (78
bytes, solo el encabezado de versión, frente a >40 000 bytes de una corrida con datos reales) y
por la ausencia de cualquier `history.json/csv` para seed 123, que esa semilla nunca llegó a
completar su primera época — no había ningún checkpoint válido que preservar. Seed 42, en
cambio, tenía summary/metadata/checkpoint/predicciones/figuras completos y coherentes
(`stopping_epoch − best_epoch = 15`, exactamente el patience configurado), evidencia de que había
terminado normalmente por early stopping antes del reinicio, no a mitad de una época. Se
eliminaron únicamente los dos artefactos vacíos de seed 123 (la carpeta de checkpoint sin pesos
y el stub de TensorBoard de 78 bytes) y se relanzó el script, que saltó automáticamente seed 42
(su `summary.json` ya existía — el script se niega a reconstruir/reentrenar una semilla cuyo
summary final ya está en disco, el mismo mecanismo de protección ya usado en R1) y entrenó
seed 123 y seed 2026 desde cero. Ningún archivo de seed 42 ni de R1 fue tocado en este proceso —
verificado comparando timestamps de archivo antes/después.

**Por qué esto no comprometía la reproducibilidad de seed 42:** `model.save_weights(...).h5`
guarda únicamente los pesos del modelo (todo lo creado vía `add_weight` dentro de las capas de
la STN) — no guarda el estado del optimizador Adam, el contador interno de early stopping, ni el
estado del generador estocástico `tf.random.Generator` de `SparseAttention`
(`self._rng` en `src/models/sparse_attention.py`, un atributo Python simple, no registrado como
peso de Keras). Esto significa que, en general, un checkpoint de este proyecto permite
**restaurar pesos** pero no **reanudar una corrida de forma bit-idéntica** desde la época exacta
donde iba — irrelevante para seed 42 (que sí terminó completa antes del reinicio), pero es la
razón documentada por la que seed 123 tuvo que reiniciarse desde cero en vez de intentar
"continuar" desde un estado a medio entrenar.

### Regla de emparejamiento seed-a-seed para la transferencia futura

Los tres checkpoints SOURCE de R2 se conservan **todos**, para emparejarse por número de semilla
con las futuras corridas locales sobre Fontibón (Config B/C/D, no implementadas todavía):
semilla local 42 ← R2 source seed 42; semilla local 123 ← R2 source seed 123; semilla local 2026
← R2 source seed 2026. Deliberadamente **no** se selecciona el checkpoint source con menor RMSE
de validación para inicializar todas las corridas locales — eso introduciría una selección
adicional basada en resultados de validación de R2, contaminando la comparación A/B/C/D con una
decisión ya sesgada hacia una semilla particular. El `output_head` nunca se transfiere en ningún
caso: se reinicializa siempre para el dominio destino (`output_head_transfer_policy: "reinitialize"`
en cada `R2_source_seed{seed}_metadata.json`), porque la escala/dinámica de PM2.5 en Fontibón es
muy distinta a la de Beijing y una cabeza de salida entrenada sobre magnitudes de Beijing
(media≈98.6 µg/m³) no tiene razón de ser útil para Fontibón (media≈18.3 µg/m³).

Compatibilidad estructural reverificada **desde el checkpoint en disco** (no desde el modelo en
memoria recién entrenado) para las tres semillas — `compatible: true` en las tres
(`R2_source_seed{42,123,2026}_summary.json`, campo `compatibility_report_from_checkpoint`).

## Fontibón STN pipeline — target de la transferencia

`src/data/fontibon.py::prepare_fontibon_stn` produce el mismo contrato (`PreparedSplit`, ahora
en `src/data/common.py` — ver nota de refactor más abajo) que R1/R2, sobre el split 70/15/15 ya
aprobado y usado por la línea base Ridge (`chronological_cuts`, `baseline_fontibon_ridge.json`):
train=10802, validation=2564, test=2574 ventanas, verificado exactamente igual tras agregar
escalado y construcción del decoder (el escalado no cambia qué filas son NaN ni qué ventanas
pasan la validación causal de `build_windows`, así que los conteos no pueden cambiar).

**Escalador propio de Fontibón, nunca el de Beijing.** `FontibonSTNScalers` se ajusta
exclusivamente con Fontibón train (nunca con Beijing). La compatibilidad de transferencia entre
Beijing-R2 y Fontibón es de forma/orden de canales (ya garantizada por
`TRANSFER_FEATURE_NAMES`), no de estadística de escalado: Beijing PM2.5 train tiene media
≈99.0 µg/m³ (std ≈91.7, máximo 994) mientras que Fontibón train tiene media ≈18.9 µg/m³
(std ≈10.8, máximo ≈100.8) — un factor ≈5 de diferencia de nivel y dispersión (ver
`outputs/metrics/domain_distance_analysis.json`). Reusar el escalador de Beijing comprimiría
casi toda la señal de Fontibón en un rango diminuto de [0,1]. Verificado con un test dedicado
(`tests/test_fontibon_stn_pipeline.py::test_scaler_is_fontibons_own_not_beijings`).

### Refactor: `PreparedSplit` movido a `src/data/common.py`

`PreparedSplit` vivía originalmente en `src/data/beijing.py`. Como `beijing.py` ya importa
`build_windows` desde `fontibon.py`, hacer que `fontibon.py` importara `PreparedSplit` de vuelta
desde `beijing.py` habría creado un import circular. Se extrajo a un módulo neutral
(`src/data/common.py`), y `beijing.py` lo re-exporta sin cambios (`from src.data.common import
PreparedSplit`), de modo que el código existente que hace
`from src.data.beijing import PreparedSplit` (p. ej. `tests/test_trainer.py`) sigue funcionando
sin modificación. Verificado: la suite completa de tests de Beijing R1/R2 y del trainer sigue
en verde después del refactor.

## Infraestructura de transferencia A/B/C/D — `src/training/transfer.py`

Construida y probada (22 tests, arquitecturas/datos sintéticos — nunca Beijing/Fontibón reales,
nunca ejecutada contra datos reales). No modifica `src/models/stn.py` (la arquitectura congelada
no se toca): `TransferSTNModel` es una subclase en un módulo aparte.

### Congelamiento con modo de inferencia forzado para submódulos congelados

`.trainable = False` en Keras solo excluye las variables de `model.trainable_variables` (el
optimizador nunca las actualiza) — **no** cambia el comportamiento del forward pass: una capa
"congelada" llamada con `training=True` seguiría aplicando Dropout estocástico y muestreo
estocástico de `SparseAttention` sobre pesos que de todas formas no van a cambiar, lo cual no es
el comportamiento esperado de un "extractor de características fijo" (Config B/C). Se resolvió
con `TransferSTNModel.call`: fuerza `training=False` en `self.encoder`/`self.decoder`
específicamente cuando ese submódulo está marcado como congelado, sin importar el `training`
que reciba el modelo completo; los submódulos no congelados siguen el `training` normal. Esto es
ortogonal a `.trainable`: uno controla qué actualiza el optimizador, el otro controla qué hace
el forward pass. Verificado en `tests/test_transfer_infrastructure.py` (pesos de encoder/decoder
exactamente sin cambios tras un paso de entrenamiento en Config B/C; los pesos no congelados sí
cambian).

### Carga selectiva de pesos — `output_head` nunca se transfiere

`load_r2_source_into_target` construye un modelo temporal con la arquitectura completa, carga
el checkpoint R2 completo en él (es la única forma de restaurar un `.h5` guardado con
`model.save_weights`), verifica compatibilidad vía `compare_transfer_weight_shapes` (ya existente
y probado, ver más arriba), y copia **únicamente** `.encoder.get_weights()` /
`.decoder.get_weights()` al modelo target — el `output_head` del modelo temporal se descarta sin
tocar nunca el `output_head` (ya inicializado de forma independiente) del modelo target.
Verificado: tras cargar, dos modelos target con semillas distintas comparten exactamente el
mismo encoder/decoder (copiado del mismo checkpoint) pero tienen `output_head` distintos entre
sí y distintos del `output_head` del propio checkpoint fuente.

### `DifferentialLRTrainer` — solo para Config D

Config A/B/C usan `Trainer` sin ningún cambio: el congelamiento es una propiedad del modelo
(`.trainable=False`), ya transparente para `Trainer._train_step` (que calcula gradientes contra
`self.model.trainable_variables`, que Keras ya filtra). Solo Config D necesita algo distinto
(tasa de aprendizaje distinta para encoder que para decoder+cabeza, con nada congelado) — se
implementó como una subclase, `DifferentialLRTrainer(Trainer)`, con dos optimizadores Adam
independientes y un `_train_step` sobreescrito que reparte los gradientes por grupo de
variables. **Limitación documentada, no oculta:** el historial por época de `Trainer.fit`
(heredado sin cambios) registra un único campo `"learning_rate"`, leído de `self.optimizer`
— para esta subclase, ese es el grupo decoder+cabeza, no el del encoder (que es fijo durante
toda la corrida y ya queda registrado una vez en la metadata/config del experimento). Se decidió
no sobreescribir el método `fit()` completo (~100 líneas) solo para añadir una columna que nunca
cambia época a época.

### Corrección de rutas de checkpoint en `configs/transfer_{B,C,D}_*.yaml`

Al auditar el estado real del proyecto (no solo los JSON de resumen) se encontró que los tres
archivos de configuración de transferencia apuntaban a
`outputs/checkpoints/r2_beijing_transfer_source/` — una ruta que **nunca existió** (minúsculas,
sin `{seed}`, un único directorio compartido entre las tres semillas). El script real
(`scripts/run_r2_source_training.py`) escribe en
`outputs/checkpoints/R2_source_seed{42,123,2026}/best.weights.h5`, uno por semilla. Se corrigió
en los tres archivos a `outputs/checkpoints/R2_source_seed{seed}/best.weights.h5` (con el
placeholder `{seed}` explícito), y `src/training/transfer.py::r2_source_checkpoint_path(seed)`
es ahora la única función que debe usarse para resolver esa ruta, precisamente para que este
tipo de error no se repita en el futuro. También se corrigió el mismo tipo de campo en
`configs/beijing_transfer_source_r2.yaml` (`output.checkpoint_root`, `status`,
`full_training.status`), que seguían describiendo R2 como "pilot_not_final"/"NOT_STARTED"
después de que el entrenamiento completo de las tres semillas ya había terminado.

## Nota sobre la partición de validación

El paper usa una partición de validación **aleatoria** dentro del conjunto de entrenamiento;
la Primera Entrega ya identificó esto como una limitación metodológica del paper (mezcla
observaciones temporalmente cercanas). Nuestra implementación usará **partición cronológica**
tanto para Beijing como para Fontibón, como adaptación deliberada — ver
`docs/PHASE2_PLAN.md`, sección "Validación temporal en Beijing".

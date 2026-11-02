# Generación de código Python con Transformers — aporte de Juan Felipe Rodríguez

## Integrantes del equipo

Equipo 21:

1. Juan Javier Valera Mejias
2. Juan David Villalba Guavita
3. Juan Felipe Rodriguez Granados
4. Leidy Dayana Vega Medina
5. Cesar Steven Leal Perez

## Estado del aporte

El notebook fue ejecutado de principio a fin en Google Colab con una GPU T4,
sin errores. La ejecución completa tomó cerca de 1 hora y 45 minutos.

Antes de fijar los modelos se hicieron 43 corridas, cuyos resultados están en
la carpeta `experimentos/`. Las decisiones se tomaron con validación, y el
conjunto de prueba solo se usó en la evaluación final del notebook.

## Objetivo

Entrenar y comparar modelos de lenguaje para generar código Python a partir
del dataset `codeparrot/codeparrot-clean`. Se comparan cinco configuraciones:

1. GPT decoder-only implementado desde cero en PyTorch.
2. `HuggingFaceTB/SmolLM2-135M` pre-entrenado.
3. `HuggingFaceTB/SmolLM2-135M` con finetuning completo.
4. `HuggingFaceTB/SmolLM2-360M` pre-entrenado.
5. `HuggingFaceTB/SmolLM2-360M` con finetuning completo.

La métrica principal es la perplejidad por token sobre prueba. La validez
sintáctica con `ast.parse` sobre 200 ejemplos es una métrica secundaria.

## Datos y preprocesamiento

- Dataset `codeparrot/codeparrot-clean` en modo streaming.
- Archivos de 1.000 a 2.000 caracteres, hasta completar 8.000.
- División 6.400 / 800 / 800 por archivo completo, con semilla fija.
- Tokenizador de SmolLM2 (49.152 tokens).
- Secuencias de 256 tokens, cortadas dentro de cada partición.

## Modelos y entrenamiento

El GPT tiene 8 capas, 8 cabezas de atención, dimensión 512 y dropout 0.1, para
un total de 50.5 millones de parámetros. La atención enmascarada está escrita
a mano, las posiciones son aprendidas y la capa de salida comparte pesos con
los embeddings. La función de generación también está escrita a mano y se usa
en todos los modelos, sin `model.generate()`.

| | GPT desde cero | SmolLM2-135M | SmolLM2-360M |
|---|---:|---:|---:|
| Parámetros (millones) | 50.5 | 134.5 | 361.8 |
| Tasa de aprendizaje | 6e-4 | 2e-4 | 1e-4 |
| Épocas | 6 | 2 | 2 |
| Lote efectivo | 16 | 16 | 16 |
| Tiempo de entrenamiento (min) | 13.3 | 27.9 | 46.3 |

Todos usan AdamW con weight decay de 0.01, calentamiento y decaimiento coseno,
y precisión mixta fp16. En los SmolLM2 también se usa acumulación de gradiente
(micro-batch de 4 con 4 pasos) y gradient checkpointing para que quepan en la
T4. Contra el sobreajuste se vuelve a cortar los bloques en cada época, se usa
parada temprana y se restauran los pesos del mejor paso de validación.

## Experimentos realizados

| Ronda | Qué se probó | Corridas | Mejor perplejidad en validación |
|---|---|---:|---:|
| R0 | Líneas base de los pre-entrenados y medición de tiempos | 8 | 135M 5.962, 360M 4.580 |
| R1a | GPT: arquitectura, regularización y tasa a 2 épocas | 12 | 29.235 |
| R1b | GPT: las 3 mejores a 6 épocas | 3 | 16.123 |
| R1c | GPT: posiciones sinusoidales con escala y modelo grande con tasa 6e-4 | 2 | 15.155 |
| R2a | 135M: tasa, calendario, lote y weight decay a media época | 10 | 5.150 |
| R2b | 135M: las 3 mejores a 2 épocas | 3 | 4.762 |
| R3a | 360M: tasas y dropout de atención a media época | 5 | 4.127 |

## Resultados sobre prueba

| Configuración | Perplejidad | Validez sintáctica |
|---|---:|---:|
| GPT desde cero | 14.662 | 50.0 % |
| SmolLM2-135M pre-entrenado | 6.016 | 63.0 % |
| SmolLM2-135M con finetuning | 4.676 | 62.0 % |
| SmolLM2-360M pre-entrenado | 4.601 | 60.5 % |
| SmolLM2-360M con finetuning | 3.854 | 57.5 % |

El mejor modelo fue SmolLM2-360M con finetuning. El finetuning bajó la
perplejidad 22.3 % en el 135M y 16.2 % en el 360M, pero no mejoró la validez
sintáctica. El código real cortado a la misma longitud da 58.0 %, por lo que
esos valores hay que leerlos frente a esa referencia y no frente a 100 %.

## Limitaciones

- La división es por archivo y no por repositorio, así que archivos de un mismo
  proyecto pueden quedar en entrenamiento y en prueba.
- Con 200 ejemplos, la validez tiene un margen cercano a 7 puntos, y los cuatro
  SmolLM2 no se pueden separar.
- Los SmolLM2 se entrenaron sin dropout. La corrida con dropout de atención en
  el 360M dio NaN con fp16 y no se investigó la causa.
- En el 360M solo se compararon tasas a media época. La corrida de 2 épocas se
  hizo una sola vez, con la tasa elegida.
- Se usó una sola semilla.

## Archivos de esta versión

| Archivo o carpeta | Contenido |
|---|---|
| `MAPLN_Microproyecto_2_JuanFelipeRodriguez.ipynb` | Notebook ejecutado, con las 10 secciones del enunciado |
| `resultados/resultados_prueba.json` | Métricas finales de las cinco configuraciones |
| `resultados/*_historial.json` | Curvas de costo y datos del entrenamiento de cada modelo |
| `experimentos/codelib.py` | Datos, GPT, entrenamiento, perplejidad, generación y validez |
| `experimentos/sweeps.py` | Configuración de cada ronda |
| `experimentos/runner.py` | Ejecuta las rondas y permite retomar una sesión cortada |
| `experimentos/Experimentos_Colab.ipynb` | Notebook para correr las rondas en Colab |
| `experimentos/resumen.csv` | Una fila por corrida con sus métricas |
| `experimentos/progreso.txt` | Registro de las corridas terminadas |
| `experimentos/runs/` | Log y resultado de cada corrida |

Los pesos de los modelos no están en este repositorio porque superan el límite
de 100 MB de GitHub. Están en el zip de la entrega: `gpt_desde_cero.pt`,
`smollm2_135m_finetuned/` y `smollm2_360m_finetuned/`.

## Ejecución

Abrir el notebook en Google Colab, elegir una GPU T4 y ejecutar todas las
celdas. El notebook descarga el dataset y los modelos, y guarda los checkpoints
en una carpeta `checkpoints/`. Si no encuentra la carpeta del proyecto en Drive,
la crea en el directorio de trabajo de la sesión.

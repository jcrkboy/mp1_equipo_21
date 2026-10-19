# Experimentos — Conv-LSTM fish

Esta carpeta contiene los scripts y los resultados de las 39 configuraciones
que se compararon para elegir la arquitectura, la función de pérdida y los
hiperparámetros del notebook `MAPLN_Microproyecto_1_JuanFelipeRodriguez.ipynb`.

Todas las corridas usan el mismo split (16.000 / 2.000 / 2.000, semilla 42),
el mismo preprocesamiento y las mismas métricas por punto que el notebook.
Las decisiones se tomaron solo con validación; el conjunto de prueba se
evalúa una única vez, dentro del notebook.

El resumen de los resultados por ronda está en el README de la carpeta
superior y en la sección 7.2 del notebook.

## Archivos

| Archivo | Contenido |
|---|---|
| `sketchlib.py` | Datos, modelos (RNN, GRU, LSTM, Conv-LSTM, Transformer causal, TCN causal, MLP de ventana), cabezas de salida (MSE, Gaussiana, mezcla de Gaussianas), pérdida enmascarada, entrenamiento, métricas y generación |
| `sweeps.py` | Configuraciones de cada ronda (`r1`, `r2`, `r3`, `r3tcn`) |
| `run_sweep.py` | Ejecuta una ronda en varios procesos en paralelo |
| `aggregate.py`, `export_results.py` | Juntan los resultados de las corridas en un CSV |
| `gen_analysis.py`, `prior_check.py` | Revisan la generación con distintas temperaturas y la corrección de priors |
| `checks.py`, `checks2.py` | Pruebas del código: pares entrada–objetivo, máscara, NLL de la mezcla contra `torch.distributions`, causalidad y equivalencia entre procesar la secuencia completa o paso a paso |
| `diagram.py` | Genera el diagrama de la arquitectura |
| `resultados/experimentos.csv` | Una fila por corrida con sus métricas de validación |
| `resultados/json/` | Configuración, historial por época y métricas de cada corrida |
| `resultados/muestras_por_corrida/` | 25 bocetos generados por cada corrida (temperatura 0.5) |
| `resultados/r*.log` | Registro de cada ronda |

## Rondas

| Ronda | Corridas | Entrenamiento | Qué se evaluó |
|---|---:|---|---|
| r1 | 13 | 15 épocas, decaimiento coseno | Familias de modelo y forma de la salida |
| r2 | 21 | 15 épocas, decaimiento coseno | Variantes del modelo recurrente, factores de la pérdida y del entrenamiento, tamaño de contexto |
| r3 y r3tcn | 5 | Hasta 60 épocas, `ReduceLROnPlateau` y parada temprana | Candidatos finales y regularización |

## Ejecución

Desde esta carpeta, con `fish.npz` dentro de una carpeta `datos/`:

```bash
python run_sweep.py --sweep r1 --data datos --out runs --workers 3 --threads 4
```

Cada corrida guarda en `runs/<nombre>/` su `result.json`, `log.txt`,
`samples.png` y `model.pt`. Las corridas que ya tienen resultados se omiten,
así que una ronda interrumpida se puede retomar con el mismo comando.

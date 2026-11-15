# Microproyecto 3 — Aporte de Juan David Villalba

## Objetivo

Comparar dos modelos de lenguaje en generación condicional de código
sobre MBPP, antes y después de un ajuste mediante LoRA.

## Archivo principal

`Microproyecto3_Equipo_21.ipynb`

El notebook conserva el código, las salidas de ejecución, las curvas
de entrenamiento y validación, y el análisis de resultados.

## Modelos

- Qwen/Qwen2.5-1.5B
- HuggingFaceTB/SmolLM2-1.7B

Se evalúa cada modelo original y su versión ajustada mediante LoRA.

## Datos y evaluación

- Dataset: MBPP, configuración full.
- Entrenamiento: 374 problemas.
- Validación: 90 problemas.
- Prueba: 500 problemas.
- Una solución por problema mediante decodificación greedy.
- Métrica: pass@1, considerando correcta una solución únicamente
  cuando pasa todos los asserts asociados.
- Mismo protocolo de generación y evaluación en las cuatro configuraciones.

## Resultados

| Modelo | Zero-shot | LoRA | Cambio |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 40.40% | 42.20% | +1.80 puntos porcentuales |
| SmolLM2-1.7B | 0.00% | 30.60% | +30.60 puntos porcentuales |

Qwen obtuvo el mayor pass@1 tanto en su versión original como ajustada.

SmolLM original produjo 499 respuestas con errores de sintaxis.
Después del ajuste, estos errores se redujeron a 2.
Por tanto, parte de su mejora corresponde a la adaptación al formato
de código válido bajo el prompt utilizado.

## Ejecución

Abrir el notebook en Google Colab con GPU T4 y ejecutar las celdas
en orden. El notebook contiene las dependencias, la configuración
y el procedimiento de respaldo y restauración en Google Drive.

Los tests ejecutan código generado. Utilizar un runtime dedicado,
sin secretos ni archivos sensibles. Drive se desmonta antes de
ejecutar las soluciones.

## Checkpoints y respaldo

Los dos adaptadores LoRA y el paquete completo se conservan en
Google Drive y no se incluyen en esta carpeta del repositorio.

Acceso para el equipo: pendiente de añadir el enlace de Drive.

## Estado del aporte

Propuesta individual para revisión del equipo.
No implica que esta sea la versión final seleccionada para Coursera.

# Microproyecto 2 — Generación de código Python con Transformers
 
Grupo: 21  
Curso: Modelos Avanzados de Procesamiento de Lenguaje Natural

## Objetivo

Comparar un modelo GPT decoder-only implementado en PyTorch con
SmolLM2-135M y SmolLM2-360M, tanto en sus versiones originales como
después de un finetuning completo sobre código Python.

## Archivo principal

`Microproyecto2_JuanDavidVillalba_Entrega.ipynb`

El notebook contiene la preparación del corpus, la implementación del GPT,
los finetunings, la evaluación comparativa, nueve ejemplos cualitativos
y las conclusiones.

## Datos y configuración

- Dataset: `codeparrot/codeparrot-clean`.
- Archivos: 8.000, filtrados entre 1.000 y 2.000 caracteres.
- División por archivos: 6.400 de entrenamiento, 800 de validación
  y 800 de prueba.
- Longitud de contexto: 256 tokens.
- Tokenizador compartido: SmolLM2, con vocabulario de 49.152 tokens.
- Finetuning completo de ambos modelos, sin LoRA ni QLoRA.
- Generación mediante un bucle autorregresivo propio, sin usar
  funciones de generación de alto nivel.

## Resultados sobre prueba

| Configuración | Perplejidad | Validez sintáctica |
|---|---:|---:|
| GPT desde cero | 33.3032 | 64.0% |
| SmolLM2-135M original | 5.7053 | 71.0% |
| SmolLM2-135M finetuned | 4.7423 | 73.0% |
| SmolLM2-360M original | 4.3929 | 71.0% |
| SmolLM2-360M finetuned | 3.8359 | 73.5% |

La perplejidad se calculó sobre 348.075 tokens objetivo.
La validez sintáctica se evaluó sobre los mismos 200 prompts
para cada configuración.

La aceptación por `ast.parse` no garantiza corrección funcional.
Los ejemplos muestran, entre otras limitaciones, repeticiones y
continuaciones de cabeceras de licencia.

## Validación de ejecución

El notebook se ejecutó de principio a fin en Google Colab con GPU
Tesla T4. Las 35 celdas de código se ejecutaron en orden y no quedaron
errores registrados.

Los checkpoints conservados corresponden a la ejecución original.
Se verificó que los tres archivos pueden abrirse y contienen
los metadatos de validación esperados.

## Reproducción

1. Abrir el notebook en Google Colab.
2. Activar una GPU CUDA.
3. Ejecutar las celdas en orden.

Ejecutar todo el notebook vuelve a entrenar los tres modelos
y recalcula las métricas. Las salidas existentes pueden consultarse
sin volver a ejecutar.

La opción `USE_DRIVE = False` mantiene los respaldos dentro del
entorno local. Para respaldar en Google Drive debe activarse
la opción correspondiente y autorizar el montaje.

## Checkpoints

Los pesos se conservan en Google Drive y no están incluidos
en esta carpeta del repositorio:

- `gpt_desde_cero_corregido.pt`
- `smollm2_135m_finetuned_corregido.pt`
- `smollm2_360m_finetuned_corregido.pt`

El enlace compartido se añadirá cuando se prepare el paquete
que revisará el grupo.

## Estado

Versión individual para revisión del equipo.
La selección de la entrega final en Coursera se realizará en grupo.

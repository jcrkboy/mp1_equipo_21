# Baseline apple — aporte de Juan David Villalba

## Estado del aporte

El notebook fue ejecutado de principio a fin en Google Colab con
GPU Tesla T4, sin errores en la ejecución verificada.

No se han realizado experimentos adicionales después de la
evaluación de prueba reportada.

## Objetivo

Generar bocetos de la categoría apple de Quick, Draw! mediante
un modelo recurrente entrenado desde cero en PyTorch.

## Datos y preprocesamiento

- Una única categoría: apple.
- Entrenamiento: 16.000 bocetos del conjunto train original.
- Validación: 2.000 bocetos del conjunto valid original.
- Prueba: 2.000 bocetos del conjunto test original.
- Selección aleatoria sin reemplazo, con semilla 42.
- Normalización calculada exclusivamente con entrenamiento.
- Conversión de stroke-3 a stroke-5.
- Hasta 199 movimientos y una señal de fin por secuencia.
- Relleno hasta 200 posiciones, excluido mediante máscaras.

Ningún boceto seleccionado necesitó truncado: la longitud máxima,
incluyendo la señal de fin, fue de 63 posiciones en los tres conjuntos.

## Arquitectura

- LSTM unidireccional de una capa.
- Entrada de dimensión 5.
- Estado oculto de dimensión 128.
- Cabeza Linear(128, 2) para desplazamientos normalizados.
- Cabeza Linear(128, 3) para logits del estado del lápiz.
- 69.765 parámetros entrenables.

Pérdida: MSE de coordenadas más entropía cruzada de estados.

La regresión utiliza movimientos reales. La clasificación utiliza
movimientos reales y la primera señal de fin. Ambas excluyen el relleno.

## Entrenamiento

- Adam, tasa de aprendizaje 0.001.
- Weight decay: 0.0001.
- Lotes de 128 bocetos.
- Máximo de 20 épocas.
- Recorte de gradientes con norma máxima 1.0.
- Parada temprana con paciencia 5.
- Selección del modelo por menor pérdida de validación.

La mejor época fue la 20 y la parada temprana no se activó.

## Resultados de la ejecución verificada

| Métrica | Resultado |
|---|---:|
| Mejor pérdida de validación | 0.559098 |
| MSE global de prueba, coordenadas normalizadas | 0.361518 |
| F1 macro de prueba, tres estados | 0.614002 |
| Entropía cruzada de prueba | 0.206157 |
| Pérdida total de prueba | 0.567674 |

La evaluación de prueba utilizó 2.000 bocetos, 75.836 movimientos
reales y 77.836 estados válidos, incluyendo una señal de fin por boceto.

Las métricas se agregaron globalmente por punto, usando los puntos
reales anteriores como contexto.

## Limitaciones

El F1 por estado fue:

- Dibujar: 0.9567.
- Levantar: 0.1464.
- Terminar: 0.7389.

La principal debilidad es la detección del estado levantar.
Las generaciones presentan formas reconocibles de manzana,
pero algunas tienen trazos abiertos y siluetas similares.

No se realizó una búsqueda sistemática de hiperparámetros
ni una evaluación formal de diversidad o memorización.

## Archivos de esta versión

- MAPLN_Microproyecto_1_JuanDavidVillalba.ipynb
- diagrama_arquitectura.png
- boceto_apple_01.png
- boceto_apple_02.png
- boceto_apple_03.png

Los tres bocetos corresponden a las semillas 101, 102 y 103,
con temperatura 1.0 y 30, 47 y 44 movimientos, respectivamente.

## Ejecución

1. Descargar el notebook y abrirlo en Google Colab.
2. Seleccionar un entorno con GPU.
3. Ejecutar todas las celdas en orden desde el principio.

El notebook descarga el dataset público y entrena el modelo desde cero.
No necesita pesos preentrenados ni acceso al Drive del autor.

Si Drive está montado, guarda allí los resultados. En caso contrario,
los guarda temporalmente en /content.

Las cifras anteriores describen la ejecución verificada; una nueva
ejecución puede producir pequeñas variaciones.

# MP2 - Versión de Cesar Leal

Notebook ejecutado completo en Colab (T4, ~2 h): `MP2_Equipo21.ipynb`

Zip de entrega con los 3 checkpoints (1,87 GB): https://drive.google.com/file/d/1fu74eKqS8WyS-JZ_OQi3iAljdhC4X8D9/view?usp=drive_link

## Resultados en prueba

| Configuración | PPL | Validez (`ast.parse`, 200) |
|---|---|---|
| GPT desde cero (44,2M) | 14,51 | 117 (58,5 %) |
| SmolLM2-135M pre-entrenado | 5,79 | 122 (61,0 %) |
| SmolLM2-135M finetuned | 4,77 | 136 (68,0 %) |
| SmolLM2-360M pre-entrenado | 4,45 | 142 (71,0 %) |
| SmolLM2-360M finetuned | 3,88 | 134 (67,0 %) |
| Referencia: código real | - | 132 (66,0 %) |

## Comparación con la versión de Juan Valera

Los splits son distintos (semilla 21 vs. 42), así que la PPL absoluta no es comparable directamente: los mismos modelos pre-entrenados dan 5,79 vs. 6,02 (135M) y 4,45 vs. 4,60 (360M). Se compara la mejora relativa al baseline:

| | Cesar | Juan |
|---|---|---|
| Finetuning 135M (cambio de PPL) | -17,6 % | -16,0 % |
| Finetuning 360M (cambio de PPL) | -12,8 % | -11,7 % |
| GPT desde cero / 135M pre-entrenado | 2,5x | 6,1x |

## Qué cambia frente a la versión de Juan

- GPT desde cero más grande (6 capas, dimensión 512, 8 cabezas, weight tying) y entrenado hasta 12 épocas con parada temprana; se detuvo en la época 10 con el mejor punto en el paso 3.033.
- Finetuning de 2 épocas en lugar de 1, guardando el mejor estado según validación.
- Checkpoints e historiales en Drive: si la sesión se cae, al volver a ejecutar se cargan y no se reentrena.
- El análisis comparativo y las conclusiones se generan a partir de los resultados de la ejecución.
- Intervalos de confianza de Wilson para la validez sintáctica.

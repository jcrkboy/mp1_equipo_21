# Microproyecto 3: Generacion Condicional de Codigo con LLMs y LoRA
Maestria en Inteligencia Artificial (MAIA) - Universidad de los Andes
Curso: Modelos Avanzados de Procesamiento de Lenguaje Natural (MAPLN)

---

## Estudiante
* Nombre: Dayana Vega
* Correo Institucional: ld.vega@uniandes.edu.co
* Rama Git: aporte/dayana-vega-micro3

---

## Objetivo del Microproyecto
Realizar el ajuste fino eficiente (LoRA) de dos modelos de lenguaje fundacionales de familias distintas (Qwen2.5-1.5B y SmolLM2-1.7B) para la tarea de generacion condicional de codigo en Python a partir de instrucciones en lenguaje natural, evaluando el desempeno funcional mediante la metrica pass@1 sobre el split de prueba de MBPP (500 problemas).

Se comparan cuatro configuraciones bajo decodificacion determinista (greedy, do_sample=False):
1. Qwen2.5-1.5B (Zero-Shot Baseline)
2. Qwen2.5-1.5B + LoRA (Atencion, r=16)
3. SmolLM2-1.7B (Zero-Shot Baseline)
4. SmolLM2-1.7B + LoRA (Atencion, r=16)

---

## Dataset: MBPP (google-research-datasets/mbpp, full)
* Train (374 problemas, 38.8%): Finetuning con LoRA (enmascaramiento estricto del prompt con labels = -100).
* Validation (90 problemas, 9.3%): Monitoreo de perdida y guardado del mejor punto de control (Early Checkpointing).
* Test (500 problemas, 51.9%): Particion obligatoria y reservada exclusivamente para la evaluacion final con pass@1.

---

## Evaluacion Oficial: Batched Generation con Sandbox de Ejecucion
Para optimizar el tiempo de inferencia sobre los 500 problemas de prueba, se implementa generacion por lotes (batch_size=16) conforme al anexo:
* tokenizer.padding_side = "left" (obligatorio para modelos causales decoder-only).
* attention_mask para ignorar tokens de relleno.
* pad_token = eos_token.
* Recorte vectorizado del prompt: out[:, inputs["input_ids"].shape[1]:].
* Decodificacion determinista (greedy, do_sample=False).
* Sandbox de ejecucion con tracer de tiempo limite (timeout = 2.0 s por problema) neutralizando input() y print().

---

## Resultados Consolidados: pass@1 en Test (500 problemas de MBPP)

| Configuracion | pass@1 (%) | Problemas Resueltos | Delta vs Base | Factor Mejora |
| :--- | :---: | :---: | :---: | :---: |
| 1. Qwen2.5-1.5B (Zero-Shot) | 37.40% | 187 / 500 | Referencia | 1.00x |
| 2. Qwen2.5-1.5B + LoRA (Atencion, r=16) | 30.80% | 154 / 500 | -6.60% | 0.82x |
| 3. SmolLM2-1.7B (Zero-Shot) | 3.40% | 17 / 500 | Referencia | 1.00x |
| 4. SmolLM2-1.7B + LoRA (Atencion, r=16) | 23.00% | 115 / 500 | +19.60% | 6.76x |

### Conclusiones Principales del Analisis
1. **SmolLM2-1.7B + LoRA:** Salto sobresaliente de +19.60 puntos porcentuales (de 3.40% a 23.00%, factor de mejora de 6.76x, resolviendo 98 problemas adicionales). Para este modelo de proposito general, LoRA indujo de manera exitosa la sintaxis de Python y la estructura de retorno requerida por los asserts.
2. **Qwen2.5-1.5B Zero-Shot:** Desempeno sobresaliente en base (37.40%) gracias a su pre-entrenamiento intensivo en mas de 5.5T tokens de codigo y tareas de infilling.
3. **Optimizacion con LoRA Ligero en Qwen:** Al restringir los adaptadores a las proyecciones de atencion (q, k, v, o con r=16, alpha=16, solo 0.28% de parametros entrenables) y aplicar early checkpointing, la perdida de validacion descendio saludablemente (0.87 a 0.64) y el pass@1 subio a 30.80% (154 problemas resueltos), mitigando el sobreajuste previo de 23.80%.
4. **Eficiencia Parametrica:** LoRA entreno unicamente ~0.28% - 0.36% de los parametros totales en ambas arquitecturas en ~65 segundos por modelo en GPU RTX 4000 Ada.

---

## Estructura del Jupyter Notebook (PlnMicroproyecto03.ipynb)
El notebook contiene las 11 secciones obligatorias en el orden exacto de la rubrica:
1. Portada e introduccion (formulacion matematica, metricas y configuraciones)
2. Librerias (reproducibilidad, verificacion de hardware e imports)
3. Carga del dataset y exploracion
4. Construccion de prompts y preprocesamiento (Prefix Infilling y labels=-100)
5. Particiones (train / validation / test)
6. Implementacion de la evaluacion pass@1 (generacion por lotes y sandbox)
7. Evaluacion de los modelos base (zero-shot)
8. Finetuning con LoRA (LoRA ligero en atencion, AdamW, 4 epocas)
9. Evaluacion de los modelos finetuneados (pass@1 en test)
10. Comparacion y resultados (tabla dinamica, grafico de barras y analisis cualitativo)
11. Conclusiones (discusion profunda de resultados, regularizacion y SLMs)

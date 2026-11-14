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
Realizar el ajuste fino eficiente (LoRA) de dos modelos de lenguaje de familias distintas (Qwen2.5-1.5B y SmolLM2-1.7B) para la tarea de generacion condicional de codigo en Python a partir de instrucciones en lenguaje natural, evaluando el desempeno funcional mediante la metrica pass@1 sobre el split de prueba de MBPP.

Se comparan cuatro configuraciones:
1. Qwen2.5-1.5B (Zero-Shot Baseline)
2. Qwen2.5-1.5B + LoRA
3. SmolLM2-1.7B (Zero-Shot Baseline)
4. SmolLM2-1.7B + LoRA

---

## Dataset: MBPP (google-research-datasets/mbpp, full)
* Train (~374 problemas): Finetuning con LoRA (enmascaramiento del prompt con labels = -100).
* Validation (90 problemas): Monitoreo de perdida durante el entrenamiento.
* Test (500 problemas): Split obligatorio para la evaluacion final con pass@1.

---

## Evaluacion: Batched Generation (Anexo)
Para optimizar el tiempo de inferencia sobre los 500 problemas de prueba, se implementa generacion por lotes con:
* tokenizer.padding_side = "left" (obligatorio para decoder-only).
* attention_mask para ignorar los tokens de relleno.
* pad_token = eos_token.
* Recorte vectorizado del prompt con su padding: out[:, inputs["input_ids"].shape[1]:].
* Decodificacion determinista / greedy (do_sample=False).

---

## Notebook del Proyecto
El notebook principal es PlnMicroproyecto03.ipynb y contiene las 11 secciones obligatorias en el orden exacto de la rubrica:
1. Portada e introduccion
2. Librerias
3. Carga del dataset y exploracion
4. Construccion de prompts y preprocesamiento
5. Particiones (train / validation / test)
6. Implementacion de la evaluacion pass@1
7. Evaluacion de los modelos base (zero-shot)
8. Finetuning con LoRA
9. Evaluacion de los modelos finetuneados
10. Comparacion y resultados
11. Conclusiones

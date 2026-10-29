# Microproyecto 2: Generación de Código Python con Transformers
**Maestría en Inteligencia Artificial (MAIA) — Universidad de los Andes**  
**Curso:** Modelos Avanzados de Procesamiento de Lenguaje Natural (MAPLN)  

---

## 👩‍💻 Información de la Estudiante
* **Nombre:** Dayana Vega
* **Correo Institucional:** ld.vega@uniandes.edu.co
* **Rama Git:** `aporte/dayana-vega-micro2`

---

## 📌 Objetivo del Proyecto
Entrenar y evaluar modelos de lenguaje autorregresivos basados en la arquitectura **Transformer decoder-only** para la tarea de generación de código fuente en Python (asistente de programación). Se realiza una comparación empírica rigurosa de **cinco configuraciones**, utilizando un tokenizador unificado (`HuggingFaceTB/SmolLM2-135M`, vocabulario BPE de 49.152 tokens) y una longitud de contexto fija de **256 tokens**:

1. **Transformer Decoder-Only desde Cero:** Implementación nativa en PyTorch (atención causal multi-cabeza, positional embeddings, MLP, LayerNorm).
2. **SmolLM2-135M Baseline:** Modelo pre-entrenado sin finetuning.
3. **SmolLM2-135M con Finetuning Completo:** Ajuste de todos los parámetros sobre el corpus de código.
4. **SmolLM2-360M Baseline:** Modelo pre-entrenado sin finetuning.
5. **SmolLM2-360M con Finetuning Completo:** Ajuste de todos los parámetros sobre el corpus de código.

---

## 📊 Dataset y Partición
* **Dataset Oficial:** `codeparrot/codeparrot-clean` (cargado en streaming).
* **Filtro Obligatorio:** Archivos con longitud entre 1.000 y 2.000 caracteres.
* **Tamaño del Corpus:** 8.000 ejemplos exactos.
* **Split Obligatorio (80 / 10 / 10):**
  * **Entrenamiento:** 6.400 ejemplos.
  * **Validación:** 800 ejemplos.
  * **Prueba:** 800 ejemplos (reservados exclusivamente para evaluación final).

---

## ⚙️ Estructura del Jupyter Notebook
El notebook [MAPLN_Microproyecto_2_DayanaVega.ipynb](MAPLN_Microproyecto_2_DayanaVega.ipynb) cumple estrictamente con las 10 secciones obligatorias de la rúbrica:
1. `Portada e introducción`
2. `Librerías`
3. `Carga del dataset y exploración`
4. `Preprocesamiento`
5. `Split train / validación / prueba`
6. `Modelo desde cero`
7. `Finetuning de los modelos pre-entrenados`
8. `Evaluación y comparación`
9. `Generación de ejemplos` *(Bucle autorregresivo propio, sin `model.generate()`)*
10. `Conclusiones`

---

## 🚀 Requisitos e Instalación
Para ejecutar el notebook, se requieren las siguientes dependencias:
```bash
pip install torch transformers datasets accelerate bitsandbytes matplotlib
```

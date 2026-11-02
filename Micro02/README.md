# Microproyecto 2: Generación de Código Python con Transformers
**Maestría en Inteligencia Artificial (MAIA) — Universidad de los Andes**  
**Curso:** Modelos Avanzados de Procesamiento de Lenguaje Natural (MAPLN)  
**Equipo:** MP1 Equipo 21  

---

## 1. Descripción y Objetivo
Entrenamiento y evaluación comparativa de modelos de lenguaje basados en la arquitectura **Transformer decoder-only** para generación de código en Python a partir del corpus oficial `codeparrot/codeparrot-clean`.

Se evalúan y contrastan rigurosamente **cinco configuraciones** bajo un tokenizador unificado (`HuggingFaceTB/SmolLM2-135M`, vocabulario de 49.152 tokens) y una longitud de contexto fija de **256 tokens**:
1. **Transformer Decoder-Only desde Cero (PyTorch)** (3 épocas, AdamW lr 3e-4)
2. **SmolLM2-135M Pre-entrenado (Baseline)**
3. **SmolLM2-135M con Finetuning Completo** (1 época, lr 5e-5)
4. **SmolLM2-360M Pre-entrenado (Baseline)**
5. **SmolLM2-360M con Finetuning Completo** (1 época, lr 3e-5)

---

## 2. Archivos en este Directorio
* **`MAPLN_Microproyecto_2_entrega.ipynb`**: Notebook principal consolidado, autocontenido y ejecutable de principio a fin (*Run All*), estructurado en las 10 secciones obligatorias de la rúbrica. Resuelve la totalidad de los requerimientos y hallazgos de evaluación (evaluación completa de las 5 configuraciones, bucle autorregresivo manual para todas las continuaciones, cálculo de perplejidad por token en test, liberación secuencial de VRAM y 3 ejemplos cualitativos por modelo entrenado).

---

## 3. Verificación de Cumplimiento de Requisitos (Rúbrica)
| Requisito / Criterio | Estado | Detalle de Implementación |
|---|---|---|
| **Filtro de longitud y tamaño** | **CUMPLE** | Filtro estricto [1.000, 2.000] caracteres; exactamente 8.000 ejemplos en streaming con caché local. |
| **Split 6.400 / 800 / 800** | **CUMPLE** | Partición 80/10/10 generada dentro del notebook con semilla fija y sin solapamiento. |
| **Contexto de 256 tokens** | **CUMPLE** | Fijo a 256 tokens en todos los modelos (GPT desde cero, baselines y finetuned). |
| **Tokenizador unificado (49.152)** | **CUMPLE** | Mismo tokenizador BPE de SmolLM2 en las 5 configuraciones para comparabilidad estricta de PPL. |
| **Finetuning Completo** | **CUMPLE** | Actualización de todos los parámetros (`all(p.requires_grad)`), sin LoRA ni QLoRA. |
| **Generación manual sin `model.generate()`** | **CUMPLE** | Bucle autorregresivo manual con logits, temperatura, top-k y softmax para todas las configuraciones. |
| **Perplejidad por token** | **CUMPLE** | $\text{PPL} = \exp(-\frac{1}{N} \sum \log p(x_i \mid x_{<i}))$ acumulada sobre los 800 archivos de prueba. |
| **Gestión de memoria VRAM** | **CUMPLE** | Carga y entrenamiento secuencial de a un modelo, liberando GPU con `gc.collect()` y `empty_cache()`. |
| **3 ejemplos por modelo entrenado** | **CUMPLE** | 3 prompts evaluados para GPT desde cero, 135M finetuned y 360M finetuned con verificación `ast.parse`. |
| **Tabla final de 5 configuraciones** | **CUMPLE** | Tabla completa con PPL, validez sintáctica (IC 95%), tiempo de entrenamiento y métricas. |


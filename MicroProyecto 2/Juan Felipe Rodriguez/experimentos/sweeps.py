"""sweeps.py - Rondas de experimentos del Microproyecto 2.

Cada corrida es un diccionario de configuracion. runner.py las ejecuta en orden, guarda el resultado en runs/
y salta las que ya terminaron, asi que una sesion de Colab cortada se retoma con otro Run All.
Todas las decisiones se toman con validacion; el conjunto de prueba no se toca en estas rondas.
"""
SMOL135, SMOL360 = "HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M"

# Rondas que ejecuta el notebook en la proxima sesion
SESION = ["R3a", "R1c"]

# Grilla de muestreo para la validez sintactica (200 archivos de validacion)
GRID_MUESTREO = [
    dict(temperature=0.0),
    dict(temperature=0.5, top_k=50, top_p=1.0),
    dict(temperature=0.7, top_k=50, top_p=1.0),
    dict(temperature=0.8, top_k=50, top_p=1.0),
    dict(temperature=1.0, top_k=50, top_p=1.0),
    dict(temperature=0.7, top_k=0, top_p=0.9),
    dict(temperature=0.8, top_k=0, top_p=0.95),
]

BASE_GPT = dict(kind="gpt", n_layer=6, n_head=8, n_embd=384, dropout=0.1, tie=True, pos="learned", epochs=2.0, lr=3e-4,
                micro_batch=16, grad_accum=1, weight_decay=0.01, warmup_frac=0.05, eval_frac=0.2, clip=1.0, resegment=False,
                seed=0, eval_batch=32)
BASE_FT = dict(kind=SMOL135, epochs=1.0, lr=5e-5, micro_batch=4, grad_accum=4, weight_decay=0.01, warmup_frac=0.05,
               eval_frac=0.1, clip=1.0, resegment=False, seed=0, eval_batch=32, checkpointing=True)


def gpt(**cambios):
    return dict(BASE_GPT, **cambios)


def ft(**cambios):
    return dict(BASE_FT, **cambios)


def medir(kind, micro, updates=8):
    """Mide tiempo y memoria con el lote efectivo de 16 que se usara al entrenar."""
    return ft(kind=kind, mode="timing", micro_batch=micro, grad_accum=16 // micro, updates=updates)


ROUNDS = {
    # R0: linea base de los pre-entrenados en validacion y tiempos reales en la T4
    "R0": {
        "b135": ft(mode="baseline", sampling=True),
        "b360": ft(kind=SMOL360, mode="baseline", sampling=True),
        "t135_m2": medir(SMOL135, 2),
        "t135_m4": medir(SMOL135, 4),
        "t135_m8": medir(SMOL135, 8),
        "t360_m2": medir(SMOL360, 2),
        "t360_m4": medir(SMOL360, 4),
        "tgpt": gpt(mode="timing", updates=40),
    },
    # R1a: GPT desde cero, tamizado a 2 epocas (arquitectura, regularizacion, tasa y variantes)
    "R1a": {
        "g01_base": gpt(),
        "g02_chico": gpt(n_layer=4, n_embd=256),
        "g03_grande": gpt(n_layer=8, n_embd=512),
        "g04_ancho": gpt(n_layer=4, n_embd=512),
        "g05_profundo": gpt(n_layer=10, n_embd=256),
        "g06_drop02": gpt(dropout=0.2),
        "g07_drop00": gpt(dropout=0.0),
        "g08_lr6e4": gpt(lr=6e-4),
        "g09_lr1e3": gpt(lr=1e-3),
        "g10_sinusoidal": gpt(pos="sin"),
        "g11_sin_atar": gpt(tie=False),
        "g12_wd01": gpt(weight_decay=0.1),
    },
    # R1b: las 3 mejores a mas epocas, re-cortando los bloques y con parada temprana
    "R1b": {"desde": "R1a", "top": 3, "prefijo": "largo",
            "cambios": dict(epochs=6.0, resegment=True, patience_evals=8, sampling=True)},
    # R1c: posiciones sinusoidales a la escala de los embeddings (g10 se ahogo con amplitud 1) y GPT grande a mas epocas
    "R1c": {
        "g13_sin_escala": gpt(pos="sin", pos_scale=0.028),
        "g14_grande_lr6e4": gpt(n_layer=8, n_embd=512, lr=6e-4, epochs=6.0, resegment=True, patience_evals=8, sampling=True),
    },
    # R2a: finetuning del 135M, tamizado a media epoca (tasa, calendario, lote efectivo, weight decay y tasa por capa)
    "R2a": {
        "f01_lr5e5": ft(epochs=0.5),
        "f02_lr2e5": ft(epochs=0.5, lr=2e-5),
        "f03_lr1e4": ft(epochs=0.5, lr=1e-4),
        "f04_lr2e4": ft(epochs=0.5, lr=2e-4),
        "f05_lr4e4": ft(epochs=0.5, lr=4e-4),
        "f06_constante": ft(epochs=0.5, min_lr_ratio=1.0, warmup_frac=0.0),
        "f07_lote8": ft(epochs=0.5, grad_accum=2),
        "f08_lote32": ft(epochs=0.5, grad_accum=8, lr=1e-4),
        "f09_wd01": ft(epochs=0.5, weight_decay=0.1),
        "f10_llrd": ft(epochs=0.5, lr=1e-4, llrd=0.9),
    },
    # R2b: las 3 mejores a 2 epocas, re-cortando los bloques y con parada temprana
    "R2b": {"desde": "R2a", "top": 3, "prefijo": "largo",
            "cambios": dict(epochs=2.0, resegment=True, patience_evals=6, sampling=True)},
    # R3a: finetuning del 360M, tamizado a media epoca con tasas ancladas en R2 y dropout de atencion contra el sobreajuste
    "R3a": {
        "h01_lr3e5": ft(kind=SMOL360, epochs=0.5, lr=3e-5),
        "h02_lr6e5": ft(kind=SMOL360, epochs=0.5, lr=6e-5),
        "h03_lr1e4": ft(kind=SMOL360, epochs=0.5, lr=1e-4),
        "h04_lr2e4": ft(kind=SMOL360, epochs=0.5, lr=2e-4),
        "h05_atdrop01": ft(kind=SMOL360, epochs=0.5, lr=1e-4, attention_dropout=0.1),
    },
    # R3b: las 2 mejores a 2 epocas, re-cortando los bloques y con parada temprana
    "R3b": {"desde": "R3a", "top": 2, "prefijo": "largo",
            "cambios": dict(epochs=2.0, resegment=True, patience_evals=6, sampling=True)},
}

"""Analiza la generacion de un modelo entrenado para varias temperaturas.

Uso: python gen_analysis.py <data_dir> <run_dir> [n]
Escribe <run_dir>/gen_tau_<t>.png y muestra estadisticas (longitud, trazos, % cortos) vs. validacion.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sketchlib as S


def main(data_dir, run_dir, n=300):
    torch.set_num_threads(4)
    ck = torch.load(os.path.join(run_dir, "model.pt"), weights_only=False)
    cfg = ck["cfg"]
    spec = S.OutputSpec(cfg["head"], cfg["n_mix"], cfg["sigma_min"])
    model = S.build_model(cfg, spec.dim)
    model.load_state_dict(ck["state"])
    cfg = {**S.DEFAULTS, **cfg}
    D = S.get_data(data_dir, cfg)
    w = S.make_class_weights(D, cfg)
    pen_bias = None if w is None else -torch.log(w)
    real = S.real_stats(D["valid"])
    print(f"reales (validación): puntos {real['len']:.1f} (incluye fin) | trazos {real['strokes']:.2f}")
    for tau in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        for pen_t in (1.0,):
            sk, lens, by_model = S.generate(model, spec, ck["scale"], n=n, temperature=tau, pen_temperature=pen_t,
                                            seed=0, pen_bias=pen_bias)
            strokes = np.array([s[:, 2].sum() for s in sk])
            short = np.mean(lens < 10)
            print(f"tau={tau:.1f} pen_t={pen_t:.1f}: puntos {lens.mean():5.1f} | trazos {strokes.mean():4.2f} | "
                  f"<10 pts {100 * short:4.1f}% | fin por el modelo {100 * by_model.mean():5.1f}%")
            if pen_t == 1.0:
                S.save_grid(sk[:30], os.path.join(run_dir, f"gen_tau_{tau:.1f}.png"), ncol=10,
                            title=f"{cfg['name']} tau={tau}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 300)

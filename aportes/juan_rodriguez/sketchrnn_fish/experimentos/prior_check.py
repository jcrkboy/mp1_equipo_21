import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sketchlib as S
torch.set_num_threads(3)
data_dir = sys.argv[1]
for run in sys.argv[2:]:
    ck = torch.load(os.path.join(run, "model.pt"), weights_only=False)
    cfg = {**S.DEFAULTS, **ck["cfg"]}
    spec = S.OutputSpec(cfg["head"], cfg["n_mix"], cfg["sigma_min"])
    model = S.build_model(cfg, spec.dim); model.load_state_dict(ck["state"])
    D = S.get_data(data_dir, cfg)
    w = S.make_class_weights(D, cfg)
    print(cfg["name"], "pesos:", None if w is None else w.numpy().round(3))
    for label, bias in [("sin corrección", None), ("corrección -log w", None if w is None else -torch.log(w))]:
        sk, lens, bm = S.generate(model, spec, ck["scale"], n=300, temperature=0.5, seed=0, pen_bias=bias)
        print(f"   {label:18s}: puntos {lens.mean():5.1f} | trazos {np.mean([s[:, 2].sum() for s in sk]):.2f} | <10 pts {100*np.mean(lens < 10):.1f}%")
        if bias is not None:
            S.save_grid(sk[:25], os.path.join(run, "samples_prior_corrected.png"), title=cfg["name"] + " (corregido)")

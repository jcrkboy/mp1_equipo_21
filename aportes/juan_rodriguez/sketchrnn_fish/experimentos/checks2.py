import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sketchlib as S
torch.manual_seed(0); torch.set_num_threads(2)
spec = S.OutputSpec("gmm", 5)
for name, extra in [("lstm", {}), ("lstm+pos", dict(abs_pos=True)), ("lstm+conv", dict(conv_front=32)),
                    ("lstm+pos+conv", dict(abs_pos=True, conv_front=32, conv_layers=3)), ("gru+pos", dict(model="gru", abs_pos=True)), ("lstm+tcnfront", dict(conv_front=32, conv_layers=4, front_type="tcn"))]:
    cfg = {**S.DEFAULTS, "hidden": 32, **extra}
    net = S.build_model(cfg, spec.dim).eval()
    x = torch.randn(3, 40, 5)
    with torch.no_grad():
        full, _ = net(x)
        st, outs = None, []
        for t in range(40):
            o, st = net(x[:, t:t+1], st); outs.append(o)
        step = torch.cat(outs, 1)
        x2 = x.clone(); x2[:, 12:] = torch.randn(3, 28, 5); f2, _ = net(x2)
    print(f"{name:14s} paso-a-paso == completo: {torch.allclose(full, step, atol=1e-5)} | causal: {torch.allclose(full[:, :12], f2[:, :12], atol=1e-5)}")
# generacion con el modelo con extensiones
cfg = {**S.DEFAULTS, "hidden": 32, "abs_pos": True, "conv_front": 32}
net = S.build_model(cfg, spec.dim)
sk, lens, bm = S.generate(net, spec, [50.0, 35.0], n=4, temperature=0.5, seed=0)
print("generacion OK:", [s.shape for s in sk], lens)

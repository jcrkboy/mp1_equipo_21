import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sketchlib as S
torch.manual_seed(0)
D = S.prepare_data(sys.argv[1], "fish", (2000, 300, 300), 42)
tr = D["train"]; X, Y, L = tr["X"], tr["Y"], tr["L"]
# 1) desplazamiento entrada/objetivo y mascara
ok = all(torch.equal(X[i, 1:L[i]], Y[i, :L[i]-1]) for i in range(len(L)))
print("X[t+1]==Y[t] en puntos reales:", ok, "| X[:,0]==START:", bool((X[:, 0] == torch.tensor(S.START_TOKEN)).all()))
cls = Y[..., 2:].argmax(-1); m = S.length_mask(L, Y.size(1))
print("ultimo punto real es 'fin':", bool((cls[torch.arange(len(L)), L-1] == 2).all()),
      "| un solo 'fin' por boceto en la mascara:", bool(((cls == 2) & m).sum(1).eq(1).all()))
print("frecuencias de clase (train, reales):", np.round(np.bincount(cls[m].numpy(), minlength=3) / m.sum().item(), 4))
print("escala (std dx, dy):", D["scale"], "| std normalizada:", Y[..., :2][m & (cls != 2)].std(0).numpy().round(3))
# 2) reconstruccion stroke-5 -> stroke-3 = original
raw = S.take_subset(S.load_category(sys.argv[1], "fish")[0], 5, np.random.default_rng(42))
s5 = S.to_stroke5(raw[0]); s3 = S.to_stroke3(s5, np.ones(2, np.float32))
print("reconstruccion stroke-3 exacta:", np.allclose(s3, raw[0]))
# 3) NLL de la GMM vs torch.distributions
spec = S.OutputSpec("gmm", 5)
reg = torch.randn(4, 7, spec.reg_dim); xy = torch.randn(4, 7, 2)
lp, mx, my, sx, sy, rho = spec.gmm(reg)
cov = torch.stack([torch.stack([sx**2, rho*sx*sy], -1), torch.stack([rho*sx*sy, sy**2], -1)], -2)
mvn = torch.distributions.MultivariateNormal(torch.stack([mx, my], -1), covariance_matrix=cov)
mix = torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=lp), mvn)
print("NLL GMM == torch.distributions:", torch.allclose(spec.reg_loss(reg, xy), -mix.log_prob(xy), atol=1e-4))
# muestreo: media empirica ~ media analitica
g = torch.Generator().manual_seed(0); r1 = reg[0, 0].repeat(20000, 1)
smp = spec.sample(r1, 1.0, g); print("media muestral vs analitica:", smp.mean(0).numpy().round(3), spec.mean(reg[0, 0]).numpy().round(3))
# 4) causalidad: perturbar el futuro no cambia salidas pasadas
for name, cfg in [("lstm", dict(model="lstm")), ("transformer", dict(model="transformer")),
                  ("transformer_w8", dict(model="transformer", window=8)), ("tcn", dict(model="tcn")), ("mlp", dict(model="mlp"))]:
    c = {**S.DEFAULTS, **cfg}; net = S.build_model(c, spec.dim).eval()
    x = torch.randn(2, 50, 5); x2 = x.clone(); x2[:, 30:] = torch.randn(2, 20, 5)
    with torch.no_grad():
        a, _ = net(x); b, _ = net(x2)
    print(f"causal {name:15s}:", torch.allclose(a[:, :30], b[:, :30], atol=1e-5), "| futuro cambia:", not torch.allclose(a[:, 30:], b[:, 30:]))
# ventana del transformer y del MLP: perturbar antes de la ventana no cambia la salida
c = {**S.DEFAULTS, "model": "transformer", "window": 8, "layers": 1}; net = S.build_model(c, spec.dim).eval()
x = torch.randn(1, 40, 5); x2 = x.clone(); x2[:, :20] = 0
with torch.no_grad(): a, _ = net(x); b, _ = net(x2)
print("transformer window=8 (1 capa) ignora t<=t-8:", torch.allclose(a[:, 27:], b[:, 27:], atol=1e-5))
c = {**S.DEFAULTS, "model": "mlp", "context": 8}; net = S.build_model(c, spec.dim).eval()
with torch.no_grad(): a, _ = net(x); b, _ = net(x2)
print("mlp context=8 ignora t<=t-8:", torch.allclose(a[:, 27:], b[:, 27:], atol=1e-5))

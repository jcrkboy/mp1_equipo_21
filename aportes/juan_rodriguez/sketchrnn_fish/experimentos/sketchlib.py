"""
sketchlib.py - Libreria de experimentos del Microproyecto 1 (Sketch-RNN sobre Quick, Draw!).

Incluye todo lo necesario para comparar modelos de forma reproducible:
  * datos: carga del .npz, split 16k/2k/2k, stroke-3 -> stroke-5, normalizacion, truncado y
    padding a 200 puntos, mascara y pares entrada -> objetivo;
  * modelos causales: RNN, GRU, LSTM, Transformer, TCN y MLP de ventana fija (contexto c);
  * cabezas de salida para (dx, dy): MSE (determinista), Gaussiana bivariada y mezcla de
    Gaussianas (GMM, como en Sketch-RNN), siempre junto a una cabeza de 3 clases para pen_state;
  * perdida compuesta enmascarada (regresion + clasificacion), entrenamiento con early stopping;
  * metricas por punto (MSE, F1 macro, perdida), generacion autorregresiva y render a PNG.
"""
import copy
import json
import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_POINTS = 200                       # limite obligatorio de puntos por boceto
START_TOKEN = (0.0, 0.0, 1.0, 0.0, 0.0)
END_TOKEN = (0.0, 0.0, 0.0, 0.0, 1.0)
CLASS_NAMES = ("lapiz_abajo", "lapiz_arriba", "fin")

DEFAULTS = dict(
    name="run", category="fish", data_seed=42, seed=0, norm="axis",
    n_train=16000, n_valid=2000, n_test=2000,
    # arquitectura
    model="lstm", hidden=256, layers=1, dropout=0.1, in_proj=0,
    abs_pos=False, conv_front=0, conv_layers=2, conv_kernel=3, front_type="conv",  # extensiones del RNN
    d_model=128, nhead=4, ff=512, window=0,           # transformer (window=0 -> contexto completo)
    channels=128, levels=5, kernel=3,                  # tcn
    context=16, mlp_layers=2,                          # mlp de ventana fija
    # salida y perdida
    head="gmm", n_mix=20, sigma_min=0.01,
    w_reg=1.0, w_pen=1.0, mse_aux=0.0, class_weights="none", label_smoothing=0.0,
    # optimizacion
    augment=0.0, batch_size=128, lr=1e-3, weight_decay=0.0, clip=1.0,
    epochs=40, patience=6, sched="plateau", sched_patience=2, min_lr=1e-5,
    # generacion para inspeccion
    gen_n=200, gen_temp=0.5, threads=4,
)


# --------------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------------
def load_category(data_dir, category):
    path = os.path.join(data_dir, f"{category}.npz")
    if not os.path.exists(path):
        import urllib.request
        url = f"https://storage.googleapis.com/quickdraw_dataset/sketchrnn/{category}.npz"
        urllib.request.urlretrieve(url, path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.load(path, encoding="latin1", allow_pickle=True)
        return data["train"], data["valid"], data["test"]


def take_subset(arr, n, rng):
    idx = rng.choice(len(arr), size=n, replace=False)
    return [np.asarray(arr[i], dtype=np.float32) for i in idx]


def to_stroke5(s3, max_points=MAX_POINTS):
    """stroke-3 (dx, dy, pen_lift) -> stroke-5 (dx, dy, p1, p2, p3) con un punto de fin explicito.

    Se truncan los trazos a max_points-1 puntos para que, junto al punto de fin, el boceto
    tenga como maximo max_points puntos reales.
    """
    s3 = s3[: max_points - 1]
    n = len(s3)
    s5 = np.zeros((n + 1, 5), np.float32)
    s5[:n, :2] = s3[:, :2]
    s5[:n, 3] = s3[:, 2]            # p2: el lapiz se levanta despues de este punto
    s5[:n, 2] = 1 - s3[:, 2]        # p1: el lapiz sigue sobre el papel
    s5[n - 1, 2:4] = (0.0, 1.0)     # el ultimo trazo siempre se cierra (relevante si se trunco)
    s5[n, 4] = 1.0                  # p3: fin del dibujo
    return s5


def pad_and_shift(sketches5, scale, max_points=MAX_POINTS):
    """Construye entradas X = [S0, P1..P199] y objetivos Y = [P1..P200] (relleno = estado fin)."""
    n = len(sketches5)
    full = np.zeros((n, max_points + 1, 5), np.float32)
    full[:, :, 4] = 1.0
    full[:, 0] = START_TOKEN
    lengths = np.zeros(n, np.int64)
    for i, s5 in enumerate(sketches5):
        L = len(s5)
        full[i, 1:L + 1] = s5
        full[i, 1:L + 1, :2] /= scale
        lengths[i] = L
    return full[:, :-1].copy(), full[:, 1:].copy(), lengths


def prepare_data(data_dir, category, n=(16000, 2000, 2000), seed=42, norm="axis"):
    tr, va, te = load_category(data_dir, category)
    rng = np.random.default_rng(seed)
    raw = {"train": take_subset(tr, n[0], rng), "valid": take_subset(va, n[1], rng),
           "test": take_subset(te, n[2], rng)}
    s5 = {k: [to_stroke5(s) for s in v] for k, v in raw.items()}
    pts = np.concatenate([s[:-1, :2] for s in s5["train"]])      # solo puntos de trazo de train
    scale = pts.std(0) if norm == "axis" else np.full(2, pts.std())
    D = {"scale": scale.astype(np.float32)}
    for k, v in s5.items():
        X, Y, L = pad_and_shift(v, D["scale"])
        D[k] = {"X": torch.from_numpy(X), "Y": torch.from_numpy(Y), "L": torch.from_numpy(L)}
    return D


def length_mask(L, T):
    return torch.arange(T)[None, :] < L[:, None]


def iterate_batches(split, batch_size, shuffle, rng=None, pool=32):
    """Lotes agrupados por longitud; cada lote se recorta a su longitud maxima real."""
    L = split["L"].numpy()
    N = len(L)
    if shuffle:
        perm = rng.permutation(N)
        batches = []
        chunk = batch_size * pool
        for s in range(0, N, chunk):
            idx = perm[s:s + chunk]
            idx = idx[np.argsort(L[idx], kind="stable")]
            batches += [idx[j:j + batch_size] for j in range(0, len(idx), batch_size)]
        order = rng.permutation(len(batches))
        batches = [batches[i] for i in order]
    else:
        idx = np.argsort(L, kind="stable")
        batches = [idx[j:j + batch_size] for j in range(0, N, batch_size)]
    for b in batches:
        bt = torch.from_numpy(b)
        T = int(L[b].max())
        yield split["X"][bt, :T], split["Y"][bt, :T], split["L"][bt]


# --------------------------------------------------------------------------------------------
# Modelos (todos causales: la salida en t solo depende de las entradas <= t)
# --------------------------------------------------------------------------------------------
POS_SCALE = 0.25     # escala de la posicion absoluta integrada (unidades normalizadas)


class RecurrentNet(nn.Module):
    """RNN/GRU/LSTM con dos extensiones opcionales que NO cambian las entradas del modelo:

    * abs_pos: capa de integracion interna (suma acumulada de los dx, dy recibidos) que agrega la
      posicion absoluta del lapiz a cada paso;
    * conv_front: capas Conv1d causales antes de la recurrencia (rasgos locales de los ultimos puntos).
    El estado para generar paso a paso es (estado_rnn, posicion_acumulada, historial_para_convs).
    """
    stateful = True

    def __init__(self, cell, hidden, layers, dropout, out_dim, in_proj=0, abs_pos=False,
                 conv_front=0, conv_layers=2, conv_kernel=3, front_type="conv"):
        super().__init__()
        in_dim = 5 + (2 if abs_pos else 0)
        self.abs_pos = abs_pos
        self.front, self.front_ctx = None, 0
        if conv_front and front_type == "conv":          # pila simple de Conv1d causales (dilatacion 1)
            mods, c_in = [], in_dim
            for _ in range(conv_layers):
                mods += [CausalConv(c_in, conv_front, conv_kernel, 1), nn.GELU()]
                c_in = conv_front
            self.front = nn.Sequential(*mods)
            self.front_ctx = conv_layers * (conv_kernel - 1)
            in_dim = conv_front
        elif conv_front and front_type == "tcn":         # bloques residuales dilatados (TCN) antes del RNN
            self.front = nn.Sequential(nn.Conv1d(in_dim, conv_front, 1),
                                       *[TCNBlock(conv_front, conv_kernel, 2 ** i, dropout)
                                         for i in range(conv_layers)])
            self.front_ctx = 2 * (conv_kernel - 1) * (2 ** conv_layers - 1)
            in_dim = conv_front
        self.proj = None
        if in_proj:
            self.proj = nn.Sequential(nn.Linear(in_dim, in_proj), nn.GELU())
            in_dim = in_proj
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(in_dim, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, state=None):
        rnn_state, pos0, hist = (None, None, None) if state is None else state
        new_pos0 = new_hist = None
        if self.abs_pos:
            if pos0 is None:
                pos0 = x.new_zeros(x.size(0), 1, 2)
            pos = pos0 + torch.cumsum(x[..., :2], dim=1)          # posicion tras aplicar cada dx, dy
            new_pos0 = pos[:, -1:]
            x = torch.cat([x, pos * POS_SCALE], dim=-1)
        if self.front is not None:
            xin = x if hist is None else torch.cat([hist, x], dim=1)
            x = self.front(xin.transpose(1, 2)).transpose(1, 2)[:, -x.size(1):]
            new_hist = xin[:, -self.front_ctx:]
        if self.proj is not None:
            x = self.proj(x)
        h, rnn_state = self.rnn(x, rnn_state)
        return self.head(self.drop(h)), (rnn_state, new_pos0, new_hist)


class CausalTransformer(nn.Module):
    stateful = False

    def __init__(self, d_model, nhead, layers, ff, dropout, out_dim, window=0, max_len=MAX_POINTS):
        super().__init__()
        self.inp = nn.Linear(5, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)
        self.window = window

    def forward(self, x, state=None):
        T = x.size(1)
        i = torch.arange(T, device=x.device)
        mask = i[None, :] > i[:, None]                       # no mirar el futuro
        if self.window:
            mask = mask | (i[None, :] <= i[:, None] - self.window)   # solo los ultimos `window`
        h = self.inp(x) + self.pos(i)[None]
        h = self.enc(h, mask=mask)
        return self.head(self.norm(h)), None


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, c, k, d, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.c1 = CausalConv(c, c, k, d)
        self.c2 = CausalConv(c, c, k, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                    # x: (B, C, T)
        h = self.ln(x.transpose(1, 2)).transpose(1, 2)
        h = self.drop(F.gelu(self.c1(h)))
        h = self.drop(self.c2(h))
        return x + h


class TCN(nn.Module):
    stateful = False

    def __init__(self, channels, levels, kernel, dropout, out_dim):
        super().__init__()
        self.inp = nn.Conv1d(5, channels, 1)
        self.blocks = nn.ModuleList([TCNBlock(channels, kernel, 2 ** i, dropout) for i in range(levels)])
        self.ln = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, out_dim)
        self.receptive_field = 1 + 2 * (kernel - 1) * (2 ** levels - 1)

    def forward(self, x, state=None):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h.transpose(1, 2))), None


class WindowMLP(nn.Module):
    """MLP que ve exactamente los ultimos `context` puntos (contexto c del enunciado)."""
    stateful = False

    def __init__(self, context, hidden, layers, dropout, out_dim):
        super().__init__()
        self.context = context
        self.conv = nn.Conv1d(5, hidden, context)
        mlp = []
        for _ in range(layers - 1):
            mlp += [nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)]
        self.mlp = nn.Sequential(*mlp, nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, state=None):
        h = self.conv(F.pad(x.transpose(1, 2), (self.context - 1, 0))).transpose(1, 2)
        return self.head(self.mlp(h)), None


def build_model(cfg, out_dim):
    m = cfg["model"]
    if m in ("rnn", "gru", "lstm"):
        return RecurrentNet(m, cfg["hidden"], cfg["layers"], cfg["dropout"], out_dim, cfg["in_proj"],
                            cfg["abs_pos"], cfg["conv_front"], cfg["conv_layers"], cfg["conv_kernel"],
                            cfg["front_type"])
    if m == "transformer":
        return CausalTransformer(cfg["d_model"], cfg["nhead"], cfg["layers"], cfg["ff"], cfg["dropout"],
                                 out_dim, cfg["window"])
    if m == "tcn":
        return TCN(cfg["channels"], cfg["levels"], cfg["kernel"], cfg["dropout"], out_dim)
    if m == "mlp":
        return WindowMLP(cfg["context"], cfg["hidden"], cfg["mlp_layers"], cfg["dropout"], out_dim)
    raise ValueError(m)


# --------------------------------------------------------------------------------------------
# Cabezas de salida para (dx, dy) + pen_state
# --------------------------------------------------------------------------------------------
class OutputSpec:
    """Define como interpretar la salida del modelo: [parametros de (dx,dy) | 3 logits de pen_state]."""

    def __init__(self, head="gmm", n_mix=20, sigma_min=0.01):
        self.kind = "mse" if head == "mse" else "gmm"
        self.M = 1 if head == "gauss" else n_mix
        self.sigma_min = sigma_min
        self.reg_dim = 2 if self.kind == "mse" else 6 * self.M
        self.dim = self.reg_dim + 3

    def split(self, out):
        return out[..., :self.reg_dim], out[..., self.reg_dim:]

    def gmm(self, reg):
        logit_pi, mx, my, sx, sy, r = torch.split(reg, self.M, dim=-1)
        sx = self.sigma_min + F.softplus(sx)
        sy = self.sigma_min + F.softplus(sy)
        rho = 0.99 * torch.tanh(r)
        return logit_pi, mx, my, sx, sy, rho

    def reg_loss(self, reg, xy):
        """Perdida de regresion por punto: MSE o NLL de la mezcla de Gaussianas bivariadas."""
        if self.kind == "mse":
            return ((reg - xy) ** 2).mean(-1)
        logit_pi, mx, my, sx, sy, rho = self.gmm(reg)
        zx = (xy[..., :1] - mx) / sx
        zy = (xy[..., 1:] - my) / sy
        omr = 1 - rho ** 2
        z = zx ** 2 + zy ** 2 - 2 * rho * zx * zy
        log_n = (-math.log(2 * math.pi) - torch.log(sx) - torch.log(sy)
                 - 0.5 * torch.log(omr) - z / (2 * omr))
        return -torch.logsumexp(F.log_softmax(logit_pi, -1) + log_n, -1)

    def mean(self, reg):
        """Prediccion puntual de (dx, dy): la esperanza de la distribucion predicha."""
        if self.kind == "mse":
            return reg
        logit_pi, mx, my, *_ = self.gmm(reg)
        pi = F.softmax(logit_pi, -1)
        return torch.stack([(pi * mx).sum(-1), (pi * my).sum(-1)], -1)

    def sample(self, reg, temperature, gen):
        if self.kind == "mse":
            return reg.clone()
        logit_pi, mx, my, sx, sy, rho = self.gmm(reg)
        t = max(float(temperature), 1e-4)
        k = torch.multinomial(F.softmax(logit_pi / t, -1), 1, generator=gen)
        mx, my, sx, sy, rho = [v.gather(-1, k).squeeze(-1) for v in (mx, my, sx, sy, rho)]
        sx, sy = sx * math.sqrt(t), sy * math.sqrt(t)
        e = torch.randn(2, k.size(0), generator=gen)
        dx = mx + sx * e[0]
        dy = my + sy * (rho * e[0] + torch.sqrt(1 - rho ** 2) * e[1])
        return torch.stack([dx, dy], -1)


def make_class_weights(D, cfg):
    if cfg["class_weights"] == "none":
        return None
    Y, L = D["train"]["Y"], D["train"]["L"]
    counts = Y[..., 2:][length_mask(L, Y.size(1))].sum(0)
    freq = counts / counts.sum()
    w = freq ** -0.5 if cfg["class_weights"] == "sqrt" else 1.0 / freq
    w = w / (freq * w).sum()                     # peso promedio por punto = 1
    return w.float()


def point_losses(out, Y, spec, cfg, class_w):
    reg, logits = spec.split(out)
    xy, cls = Y[..., :2], Y[..., 2:].argmax(-1)
    reg_l = spec.reg_loss(reg, xy)
    pen_l = F.cross_entropy(logits.reshape(-1, 3), cls.reshape(-1), weight=class_w,
                            label_smoothing=cfg["label_smoothing"], reduction="none").view_as(reg_l)
    mse_l = ((spec.mean(reg) - xy) ** 2).mean(-1)
    return reg_l, pen_l, mse_l


def combine(cfg, spec, reg, pen, mse):
    loss = cfg["w_reg"] * reg + cfg["w_pen"] * pen
    if cfg["mse_aux"] and spec.kind != "mse":
        loss = loss + cfg["mse_aux"] * mse
    return loss


# --------------------------------------------------------------------------------------------
# Entrenamiento y evaluacion
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model, spec, split, cfg, class_w=None, batch_size=500):
    model.eval()
    sums = np.zeros(3)
    n = 0
    means, tgts, logits_all, cls_all = [], [], [], []
    for X, Y, L in iterate_batches(split, batch_size, shuffle=False):
        out, _ = model(X)
        mask = length_mask(L, X.size(1))
        reg_l, pen_l, mse_l = point_losses(out, Y, spec, cfg, class_w)
        sums += [reg_l[mask].sum().item(), pen_l[mask].sum().item(), mse_l[mask].sum().item()]
        n += int(mask.sum())
        reg, logits = spec.split(out)
        means.append(spec.mean(reg)[mask])
        tgts.append(Y[..., :2][mask])
        logits_all.append(logits[mask])
        cls_all.append(Y[..., 2:].argmax(-1)[mask])
    cat = lambda v: torch.cat(v).numpy()
    reg, pen, mse = sums / n
    return dict(mean=cat(means), xy=cat(tgts), logits=cat(logits_all), cls=cat(cls_all),
                reg_loss=reg, pen_loss=pen, mse_loss=mse, loss=combine(cfg, spec, reg, pen, mse), n=n)


def f1_scores(y, yhat, k=3):
    cm = np.bincount(y * k + yhat, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    pp, ap = cm.sum(0), cm.sum(1)
    prec = np.divide(tp, pp, out=np.zeros(k), where=pp > 0)
    rec = np.divide(tp, ap, out=np.zeros(k), where=ap > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return f1, cm


def compute_metrics(pred, scale, bias=None):
    err = pred["mean"] - pred["xy"]
    err_px = err * scale
    logits = pred["logits"] if bias is None else pred["logits"] + bias
    f1, cm = f1_scores(pred["cls"], logits.argmax(1))
    return dict(loss=float(pred["loss"]), reg_loss=float(pred["reg_loss"]), pen_loss=float(pred["pen_loss"]),
                mse_norm=float((err ** 2).mean()), mse_px=float((err_px ** 2).mean()),
                mse_x_px=float((err_px[:, 0] ** 2).mean()), mse_y_px=float((err_px[:, 1] ** 2).mean()),
                f1_macro=float(f1.mean()), f1_down=float(f1[0]), f1_up=float(f1[1]), f1_end=float(f1[2]),
                acc=float(np.trace(cm) / cm.sum()), cm=cm.tolist(), n_points=int(pred["n"]))


def tune_bias(pred, grid=np.linspace(-4, 4, 33)):
    """Ajusta sesgos aditivos de los logits (clases 1 y 2) para maximizar F1 macro en validacion."""
    best_f1, best_b = -1.0, np.zeros(3)
    for b1 in grid:
        for b2 in grid:
            b = np.array([0.0, b1, b2], np.float32)
            f1, _ = f1_scores(pred["cls"], (pred["logits"] + b).argmax(1))
            if f1.mean() > best_f1:
                best_f1, best_b = float(f1.mean()), b
    return best_f1, best_b


def train(model, spec, D, cfg, log=print):
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    class_w = make_class_weights(D, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["sched"] == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5,
                                                           patience=cfg["sched_patience"], min_lr=cfg["min_lr"])
    elif cfg["sched"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["min_lr"])
    else:
        sched = None
    best, best_state, best_ep, wait, hist = float("inf"), None, 0, 0, []
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        tot = np.zeros(4)
        for X, Y, L in iterate_batches(D["train"], cfg["batch_size"], True, rng):
            if cfg["augment"] > 0:
                f = 1 + (torch.rand(X.size(0), 1, 2) * 2 - 1) * cfg["augment"]
                X, Y = X.clone(), Y.clone()
                X[..., :2] *= f
                Y[..., :2] *= f
            out, _ = model(X)
            mask = length_mask(L, X.size(1)).float()
            n = mask.sum()
            reg_l, pen_l, mse_l = point_losses(out, Y, spec, cfg, class_w)
            reg, pen, mse = [(v * mask).sum() / n for v in (reg_l, pen_l, mse_l)]
            loss = combine(cfg, spec, reg, pen, mse)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            tot += np.array([loss.item(), reg.item(), pen.item(), 1.0]) * n.item()
        tr = tot[:3] / tot[3]
        pv = predict(model, spec, D["valid"], cfg, class_w)
        val = compute_metrics(pv, D["scale"])
        hist.append(dict(epoch=ep, train_loss=tr[0], train_reg=tr[1], train_pen=tr[2],
                         val_loss=val["loss"], val_reg=val["reg_loss"], val_pen=val["pen_loss"],
                         val_mse=val["mse_norm"], val_f1=val["f1_macro"],
                         lr=opt.param_groups[0]["lr"], secs=time.time() - t0))
        log(f"ep {ep:3d} train {tr[0]:.4f} val {val['loss']:.4f} mse {val['mse_norm']:.4f} "
            f"f1 {val['f1_macro']:.4f} lr {opt.param_groups[0]['lr']:.1e} {time.time() - t0:.1f}s")
        if sched is not None:
            sched.step(val["loss"]) if cfg["sched"] == "plateau" else sched.step()
        if val["loss"] < best - 1e-4:
            best, best_state, best_ep, wait = val["loss"], copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                log(f"early stopping (mejor epoca {best_ep})")
                break
    model.load_state_dict(best_state)
    return hist, best_ep


# --------------------------------------------------------------------------------------------
# Generacion y render
# --------------------------------------------------------------------------------------------
def to_stroke3(s5, scale):
    s5 = s5[s5[:, 4] == 0]
    s3 = np.zeros((len(s5), 3), np.float32)
    s3[:, :2] = s5[:, :2] * scale
    s3[:, 2] = s5[:, 3]
    if len(s3):
        s3[-1, 2] = 1.0
    return s3


@torch.no_grad()
def generate(model, spec, scale, n=16, temperature=0.5, pen_temperature=1.0, greedy_pen=False,
             max_points=MAX_POINTS, seed=0, pen_bias=None):
    """Generacion autorregresiva en lote. Devuelve bocetos stroke-3 (unidades originales).

    pen_bias: sesgo aditivo a los logits del lapiz al muestrear (p. ej. -log(pesos de clase) para deshacer
    el reponderado de la entropia cruzada y recuperar probabilidades calibradas).
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    x = torch.tensor(START_TOKEN).repeat(n, 1, 1)
    seq, state = x, None
    pts = torch.zeros(n, max_points, 5)
    pts[..., 4] = 1.0
    done = torch.zeros(n, dtype=torch.bool)
    by_model = torch.zeros(n, dtype=torch.bool)
    lengths = torch.full((n,), max_points)
    end = torch.tensor(END_TOKEN)
    for t in range(max_points):
        if model.stateful:
            out, state = model(x, state)
        else:
            out, _ = model(seq)
        reg, logits = spec.split(out[:, -1])
        if pen_bias is not None:
            logits = logits + pen_bias
        dxdy = spec.sample(reg, temperature, gen)
        if greedy_pen:
            cls = logits.argmax(-1)
        else:
            cls = torch.multinomial(F.softmax(logits / pen_temperature, -1), 1, generator=gen).squeeze(1)
        by_model |= (~done) & (cls == 2)
        if t == max_points - 1:
            cls = torch.full_like(cls, 2)
        p = torch.zeros(n, 5)
        p[:, :2] = dxdy
        p[torch.arange(n), 2 + cls] = 1.0
        p[cls == 2, :2] = 0.0
        p[done] = end
        pts[:, t] = p
        new_end = (~done) & (cls == 2)
        lengths[new_end] = t + 1
        done |= new_end
        if bool(done.all()):
            break
        x = p.unsqueeze(1)
        if not model.stateful:
            seq = torch.cat([seq, x], 1)
    sketches = [to_stroke3(pts[i, :lengths[i]].numpy(), scale) for i in range(n)]
    return sketches, lengths.numpy(), by_model.numpy()


def draw_sketch(ax, s3, lw=2.0, color="black"):
    xy = np.cumsum(s3[:, :2], axis=0)
    start = 0
    for i in range(len(s3)):
        if s3[i, 2] == 1:
            ax.plot(xy[start:i + 1, 0], -xy[start:i + 1, 1], color=color, linewidth=lw)
            start = i + 1
    ax.set_aspect("equal")
    ax.axis("off")


def save_grid(sketches, path, ncol=5, title=None, lw=1.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrow = int(math.ceil(len(sketches) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2 * ncol, 2 * nrow))
    for ax, s in zip(np.ravel(axes), sketches):
        draw_sketch(ax, s, lw=lw)
    for ax in np.ravel(axes)[len(sketches):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


def gen_stats(sketches, lengths, by_model):
    strokes = [int(s[:, 2].sum()) for s in sketches]
    return dict(gen_len=float(np.mean(lengths)), gen_strokes=float(np.mean(strokes)),
                gen_end_rate=float(np.mean(by_model)))


def real_stats(split):
    L = split["L"]
    mask = length_mask(L, split["Y"].size(1))
    strokes = (split["Y"][..., 3] * mask).sum(1)
    return dict(len=float(L.float().mean()), strokes=float(strokes.mean()))


# --------------------------------------------------------------------------------------------
# Experimento completo
# --------------------------------------------------------------------------------------------
_DATA_CACHE = {}


def get_data(data_dir, cfg):
    key = (data_dir, cfg["category"], cfg["n_train"], cfg["n_valid"], cfg["n_test"], cfg["data_seed"], cfg["norm"])
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = prepare_data(data_dir, cfg["category"], (cfg["n_train"], cfg["n_valid"], cfg["n_test"]),
                                        cfg["data_seed"], cfg["norm"])
    return _DATA_CACHE[key]


def run_experiment(cfg, data_dir, out_dir):
    cfg = {**DEFAULTS, **cfg}
    run_dir = os.path.join(out_dir, cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    logf = open(os.path.join(run_dir, "log.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, file=logf, flush=True)

    D = get_data(data_dir, cfg)
    spec = OutputSpec(cfg["head"], cfg["n_mix"], cfg["sigma_min"])
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg, spec.dim)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"{cfg['name']}  params={n_params}  cfg={json.dumps(cfg)}")
    t0 = time.time()
    hist, best_ep = train(model, spec, D, cfg, log)
    train_time = time.time() - t0
    class_w = make_class_weights(D, cfg)
    pv = predict(model, spec, D["valid"], cfg, class_w)
    val = compute_metrics(pv, D["scale"])
    f1_tuned, bias = tune_bias(pv)
    pen_bias = None if class_w is None else -torch.log(class_w)      # deshace el reponderado al muestrear
    sk, lens, by_model = generate(model, spec, D["scale"], n=cfg["gen_n"], temperature=cfg["gen_temp"],
                                  seed=cfg["seed"], pen_bias=pen_bias)
    gstats = gen_stats(sk, lens, by_model)
    save_grid(sk[:25], os.path.join(run_dir, "samples.png"), title=cfg["name"])
    torch.save({"state": model.state_dict(), "cfg": cfg, "scale": D["scale"]}, os.path.join(run_dir, "model.pt"))
    result = dict(name=cfg["name"], cfg=cfg, n_params=n_params, epochs=len(hist), best_epoch=best_ep,
                  train_time=train_time, val=val, val_f1_tuned=f1_tuned, bias=bias.tolist(),
                  gen=gstats, real=real_stats(D["valid"]), history=hist)
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    log(f"DONE val_loss={val['loss']:.4f} mse={val['mse_norm']:.4f} f1={val['f1_macro']:.4f} "
        f"f1_tuned={f1_tuned:.4f} gen={gstats} time={train_time:.0f}s")
    logf.close()
    return result

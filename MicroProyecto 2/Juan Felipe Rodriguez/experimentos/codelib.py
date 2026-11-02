"""codelib.py - Libreria de experimentos del Microproyecto 2 (generacion de codigo Python con Transformers).

Incluye: corpus y split por archivo, segmentacion en bloques de 256 tokens, GPT decoder-only implementado
a mano, entrenamiento comun (AMP, acumulacion de gradiente, gradient checkpointing), perplejidad por token,
generacion autorregresiva escrita a mano y validez sintactica con ast.parse.
"""
import ast
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT = 256
VOCAB_SIZE = 49152
SPLITS = (6400, 800, 800)
TOKENIZER = "HuggingFaceTB/SmolLM2-135M"


# --------------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------------
def load_corpus(cache_path, n_examples=8000, min_chars=1000, max_chars=2000):
    """Streaming de codeparrot-clean con filtro de longitud; guarda el resultado en cache_path."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return [json.loads(line)["content"] for line in f]
    from datasets import load_dataset
    stream = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    corpus = []
    for example in stream:
        code = example["content"]
        if min_chars <= len(code) <= max_chars:
            corpus.append(code)
            if len(corpus) >= n_examples:
                break
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            for code in corpus:
                f.write(json.dumps({"content": code}) + "\n")
    return corpus


def split_files(corpus, seed=42):
    """Split por archivo (6.400/800/800) ANTES de concatenar, para que ningun archivo quede en dos conjuntos."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(corpus))
    a, b = SPLITS[0], SPLITS[0] + SPLITS[1]
    return {"train": [corpus[i] for i in idx[:a]], "valid": [corpus[i] for i in idx[a:b]],
            "test": [corpus[i] for i in idx[b:b + SPLITS[2]]]}


def tokenize_files(files, tokenizer):
    """Tokeniza cada archivo y le agrega el token de fin (separa archivos al concatenar)."""
    ids = tokenizer(files, add_special_tokens=False)["input_ids"]
    return [np.array(x + [tokenizer.eos_token_id], dtype=np.int32) for x in ids]


def make_blocks(file_ids, context=CONTEXT, offset=0, rng=None):
    """Concatena los archivos (en orden aleatorio si hay rng) y corta bloques de `context` tokens sin solape."""
    order = np.arange(len(file_ids)) if rng is None else rng.permutation(len(file_ids))
    stream = np.concatenate([file_ids[i] for i in order])[offset:]
    n = len(stream) // context
    return torch.from_numpy(stream[:n * context].reshape(n, context).astype(np.int64))


# --------------------------------------------------------------------------------------------
# GPT decoder-only implementado a mano
# --------------------------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, context):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.key_query_value = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop, self.resid_drop = nn.Dropout(dropout), nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(context, context, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key_query_value(x).chunk(3, dim=-1)
        shape = (B, T, self.n_head, self.head_dim)
        k, q, v = [t.view(shape).transpose(1, 2) for t in (k, q, v)]           # (B, h, T, d)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))               # no mirar el futuro
        att = self.attn_drop(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, context):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, context)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))          # pre-LN, como GPT-2
        return x + self.mlp(self.ln2(x))


def sinusoidal(context, n_embd):
    t = torch.arange(context, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, n_embd, 2, dtype=torch.float32) * (-math.log(10000.0) / n_embd))
    table = torch.zeros(context, n_embd)
    table[:, 0::2] = torch.sin(t * freq)
    table[:, 1::2] = torch.cos(t * freq)
    return table


class GPT(nn.Module):
    def __init__(self, n_layer=6, n_head=8, n_embd=384, dropout=0.1, context=CONTEXT, vocab_size=VOCAB_SIZE, tie=True,
                 pos="learned", pos_scale=1.0):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(context, n_embd) if pos == "learned" else None
        if pos == "sin":
            self.register_buffer("pos_table", pos_scale * sinusoidal(context, n_embd), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout, context) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie:
            self.head.weight = self.tok.weight
        self.apply(self._init)
        for name, p in self.named_parameters():                     # escala las salidas residuales (GPT-2)
            if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.size(1)
        pos = self.pos_table[:T] if self.pos is None else self.pos(torch.arange(T, device=idx.device))
        x = self.drop(self.tok(idx) + pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


def logits_of(model, ids):
    """Logits (B, T, V) tanto para el GPT propio como para los modelos de Hugging Face."""
    out = model(ids)
    return out if torch.is_tensor(out) else out.logits


def count_params(model):
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())


# --------------------------------------------------------------------------------------------
# Perplejidad por token
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def token_nll(model, blocks, device, batch_size=16, amp_dtype=None):
    """Suma de -log p(x_i | x_<i) sobre todos los tokens y numero de tokens evaluados."""
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(blocks), batch_size):
        b = blocks[s:s + batch_size].to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = logits_of(model, b)
        nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), b[:, 1:].reshape(-1), reduction="sum")
        total += nll.item()
        n += b[:, 1:].numel()
    return total, n


def perplexity(model, blocks, device, batch_size=16, amp_dtype=None):
    total, n = token_nll(model, blocks, device, batch_size, amp_dtype)
    return math.exp(total / n), total / n, n


# --------------------------------------------------------------------------------------------
# Generacion autorregresiva (sin model.generate) y validez sintactica
# --------------------------------------------------------------------------------------------
def filter_logits(logits, top_k=0, top_p=1.0):
    if top_k:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        drop = cum > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        logits = logits.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return logits


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, device, eos_id, temperature=0.8, top_k=50, top_p=1.0, seed=0,
             amp_dtype=None, context=CONTEXT):
    """Bucle autorregresivo en lote: logits -> distribucion -> muestreo -> concatenar al contexto.

    prompt_ids: tensor (B, P) con prefijos de igual longitud. Devuelve (ids, longitudes, termino_por_eos).
    En los modelos de Hugging Face se reutiliza el KV-cache mientras la secuencia quepa en el contexto; el
    GPT propio vuelve a procesar la ventana de los ultimos `context` tokens en cada paso.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    ids = prompt_ids.to(device)
    B = ids.size(0)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    lengths = torch.full((B,), ids.size(1) + max_new_tokens, device=device)
    cached = hasattr(model, "config") and ids.size(1) + max_new_tokens <= context
    past = None
    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            if cached:
                out = model(input_ids=ids if past is None else ids[:, -1:], past_key_values=past, use_cache=True)
                logits, past = out.logits[:, -1].float(), out.past_key_values
            else:
                logits = logits_of(model, ids[:, -context:])[:, -1].float()
        if temperature <= 0:
            nxt = logits.argmax(-1)
        else:
            probs = F.softmax(filter_logits(logits / temperature, top_k, top_p), dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        nxt = nxt.masked_fill(done, eos_id)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        new_done = (~done) & (nxt == eos_id)
        lengths[new_done] = ids.size(1) - 1
        done |= new_done
        if bool(done.all()):
            break
    return ids.cpu(), lengths.cpu(), done.cpu()


def is_valid_python(code):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False


def strip_truncated_line(code):
    """Descarta la ultima linea cuando la generacion fue cortada por el limite de tokens."""
    return code.rsplit("\n", 1)[0] if "\n" in code else ""


def syntax_validity(model, tokenizer, files_ids, device, n=200, prompt_tokens=128, new_tokens=128, seed=0, amp_dtype=None,
                    batch_size=32, **sampling):
    """Porcentaje de continuaciones sintacticamente validas sobre n archivos (semilla fija)."""
    rng = np.random.default_rng(seed)
    pool = [f for f in files_ids if len(f) > prompt_tokens + 1]
    chosen = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    prompts = torch.tensor(np.stack([pool[i][:prompt_tokens] for i in chosen]).astype(np.int64))
    valid, ended, texts = [], [], []
    for s in range(0, len(prompts), batch_size):
        ids, lengths, done = generate(model, prompts[s:s + batch_size], new_tokens, device, tokenizer.eos_token_id,
                                      seed=seed + s, amp_dtype=amp_dtype, **sampling)
        for row, L, d in zip(ids, lengths, done):
            code = tokenizer.decode(row[:int(L)].tolist(), skip_special_tokens=True)
            if not d:
                code = strip_truncated_line(code)
            valid.append(is_valid_python(code))
            ended.append(bool(d))
            texts.append(code)
    return 100 * float(np.mean(valid)), 100 * float(np.mean(ended)), texts


def unique_line_ratio(texts):
    """Fraccion de lineas no repetidas: evita premiar un muestreo que solo copia la misma linea."""
    ratios = []
    for text in texts:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        ratios.append(len(set(lines)) / max(1, len(lines)))
    return 100 * float(np.mean(ratios))


def reference_validity(tokenizer, files_ids, n=200, total_tokens=256, seed=0):
    """Techo de la metrica: codigo REAL cortado en la misma longitud, con el mismo tratamiento."""
    rng = np.random.default_rng(seed)
    pool = [f for f in files_ids if len(f) > total_tokens // 2 + 1]
    chosen = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    out = []
    for i in chosen:
        f = pool[i][:total_tokens]
        code = tokenizer.decode(f.tolist(), skip_special_tokens=True)
        if f[-1] != tokenizer.eos_token_id:
            code = strip_truncated_line(code)
        out.append(is_valid_python(code))
    return 100 * float(np.mean(out))


# --------------------------------------------------------------------------------------------
# Entrenamiento comun
# --------------------------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_setup(device):
    """bf16 nativo (Ampere o mas reciente); si no, fp16 con GradScaler; en CPU sin precision mixta."""
    if device.type != "cuda":
        return None, None
    if torch.cuda.get_device_capability(device)[0] >= 8:        # la T4 es 7.5: su bf16 seria emulado y lento
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")


def lr_at(step, total, warmup, base_lr, min_ratio=0.0):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def layer_depth(name, n_layers):
    """0 para los embeddings, i + 1 para la capa i y n_layers + 1 para lo que va despues de las capas."""
    parts = name.split(".")
    for key in ("layers", "blocks"):
        if key in parts:
            return int(parts[parts.index(key) + 1]) + 1
    return 0 if any(t in name for t in ("embed", "tok", "pos")) else n_layers + 1


def param_groups(model, weight_decay, llrd=1.0):
    """Sin weight decay en normas y sesgos. llrd < 1: las capas cercanas a la entrada reciben una tasa menor."""
    names = [n for n, _ in model.named_parameters()]
    n_layers = 1 + max([layer_depth(n, 0) - 1 for n in names if ".layers." in n or "blocks." in n] + [0])
    groups, seen = {}, set()
    for name, p in model.named_parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        decays = p.dim() >= 2 and "ln" not in name.split(".")[-2:] and "norm" not in name
        scale = llrd ** (n_layers + 1 - layer_depth(name, n_layers))
        key = (weight_decay if decays else 0.0, scale)
        groups.setdefault(key, dict(params=[], weight_decay=key[0], lr_scale=scale))["params"].append(p)
    return list(groups.values())


def train(model, train_ids, valid_blocks, device, cfg, log=print, on_best=None):
    """Bucle comun. cfg: epochs, lr, micro_batch, grad_accum, weight_decay, warmup_frac, eval_every, patience_evals,
    resegment (re-corta los bloques en cada epoca), clip, seed, min_lr_ratio, llrd.
    """
    amp_dtype, scaler = amp_setup(device)
    rng = np.random.default_rng(cfg["seed"])
    blocks0 = make_blocks(train_ids)
    micro, accum = cfg["micro_batch"], cfg["grad_accum"]
    updates_per_epoch = max(1, len(blocks0) // (micro * accum))
    total = int(math.ceil(cfg["epochs"] * updates_per_epoch))
    warmup = int(cfg["warmup_frac"] * total)
    opt = torch.optim.AdamW(param_groups(model, cfg["weight_decay"], cfg.get("llrd", 1.0)), lr=cfg["lr"], betas=(0.9, 0.999))
    hist, step, best, best_step, bad = [], 0, float("inf"), 0, 0
    t0 = time.time()
    running, running_n = 0.0, 0
    epoch, stop = 0, False
    opt.zero_grad(set_to_none=True)
    while step < total and not stop:
        epoch += 1
        if cfg.get("resegment", False):
            blocks = make_blocks(train_ids, offset=int(rng.integers(0, CONTEXT)), rng=rng)
        else:
            blocks = blocks0
        order = torch.from_numpy(rng.permutation(len(blocks)))
        for i in range(0, len(order) - micro + 1, micro):
            model.train()
            x = blocks[order[i:i + micro]].to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = logits_of(model, x)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), x[:, 1:].reshape(-1))
            running += loss.item()
            running_n += 1
            loss = loss / accum
            (scaler.scale(loss) if scaler else loss).backward()
            if (i // micro + 1) % accum:
                continue
            lr = lr_at(step, total, warmup, cfg["lr"], cfg.get("min_lr_ratio", 0.0))
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_scale"]
            if scaler:
                scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg["eval_every"] == 0 or step == total:
                ppl, nll, _ = perplexity(model, valid_blocks, device, cfg.get("eval_batch", 16), amp_dtype)
                hist.append(dict(step=step, epoch=step / updates_per_epoch, train_loss=running / running_n, val_loss=nll,
                                 val_ppl=ppl, lr=lr, secs=time.time() - t0))
                log(f"step {step}/{total} ep {step / updates_per_epoch:.2f} train {running / running_n:.4f} "
                    f"val {nll:.4f} ppl {ppl:.3f} {time.time() - t0:.0f}s")
                running, running_n = 0.0, 0
                if nll < best - 1e-4:
                    best, best_step, bad = nll, step, 0
                    if on_best:
                        on_best(model)
                else:
                    bad += 1
                    if cfg.get("patience_evals") and bad >= cfg["patience_evals"]:
                        log(f"early stopping (mejor paso {best_step})")
                        stop = True
                        break
            if step >= total:
                break
    return hist, best, best_step

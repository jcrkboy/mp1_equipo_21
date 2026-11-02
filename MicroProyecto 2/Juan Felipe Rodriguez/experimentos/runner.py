"""runner.py - Ejecuta las rondas de sweeps.py y guarda en runs/<ronda>/<corrida>/ el log y el resultado."""
import csv
import gc
import glob
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import codelib as L
import sweeps

MONITOR_BLOCKS = 400        # bloques de validacion usados durante el entrenamiento; al final se evalua todo
N_VALIDEZ = 200


class Log:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        line = time.strftime("%H:%M:%S ") + str(msg)
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


class Datos:
    """Corpus, tokenizador y bloques. La particion de prueba no se tokeniza aqui."""

    def __init__(self, exp_dir):
        self.tok = AutoTokenizer.from_pretrained(L.TOKENIZER)
        sp = L.split_files(L.load_corpus(os.path.join(exp_dir, "corpus_8000.jsonl")))
        self.train = L.tokenize_files(sp["train"], self.tok)
        self.valid = L.tokenize_files(sp["valid"], self.tok)
        self.valid_blocks = L.make_blocks(self.valid)
        self.monitor = self.valid_blocks[:MONITOR_BLOCKS]
        self.n_train_blocks = len(L.make_blocks(self.train))
        print(f"bloques: train {self.n_train_blocks} | valid {len(self.valid_blocks)}")


def build_model(cfg, device):
    if cfg["kind"] == "gpt":
        arch = {k: cfg[k] for k in ("n_layer", "n_head", "n_embd", "dropout", "tie", "pos")}
        return L.GPT(pos_scale=cfg.get("pos_scale", 1.0), **arch).to(device)
    model = AutoModelForCausalLM.from_pretrained(cfg["kind"], attention_dropout=cfg.get("attention_dropout", 0.0))
    model = model.to(torch.float32)     # pesos maestros en fp32
    model.config.use_cache = False
    if cfg.get("checkpointing", True):
        model.gradient_checkpointing_enable()
    return model.to(device)


def sampling_grid(model, data, device, amp_dtype, log):
    rows = []
    for s in sweeps.GRID_MUESTREO:
        t0 = time.time()
        valid, ended, texts = L.syntax_validity(model, data.tok, data.valid, device, n=N_VALIDEZ, amp_dtype=amp_dtype, **s)
        rows.append(dict(s, valid=valid, ended=ended, unique_lines=L.unique_line_ratio(texts), secs=time.time() - t0,
                         samples=texts[:3]))
        log(f"muestreo {s}: validez {valid:.1f} % | eos {ended:.1f} % | lineas unicas {rows[-1]['unique_lines']:.1f} %")
    return rows


def run_one(rnd, run_id, cfg, data, out_dir, device):
    folder = os.path.join(out_dir, rnd, run_id)
    path = os.path.join(folder, "result.json")
    if os.path.exists(path):
        print(f"{rnd}/{run_id}: ya existe, se salta")
        return
    os.makedirs(folder, exist_ok=True)
    log = Log(os.path.join(folder, "log.txt"))
    log(f"inicio {rnd}/{run_id}: {cfg}")
    L.set_seed(cfg.get("seed", 0))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = build_model(cfg, device)
    res = dict(id=run_id, round=rnd, cfg=cfg, n_params=L.count_params(model))
    amp_dtype, _ = L.amp_setup(device)
    mode = cfg.get("mode", "train")
    updates_epoch = data.n_train_blocks // (cfg["micro_batch"] * cfg["grad_accum"])

    if mode == "timing":
        tcfg = dict(cfg, epochs=cfg["updates"] / updates_epoch, eval_every=cfg["updates"])
        hist, _, _ = L.train(model, data.train, data.monitor[:4], device, tcfg, log=log)
        per_block = hist[-1]["secs"] / (cfg["updates"] * cfg["micro_batch"] * cfg["grad_accum"])
        res.update(secs_per_block=per_block, min_per_epoch=per_block * data.n_train_blocks / 60)
    elif mode == "train":
        best_state = {}

        def keep_best(m):
            best_state.update({k: v.detach().cpu().clone() for k, v in m.state_dict().items()})

        tcfg = dict(cfg, eval_every=max(1, round(cfg["eval_frac"] * updates_epoch)))
        hist, best, best_step = L.train(model, data.train, data.monitor, device, tcfg, log=log, on_best=keep_best)
        res.update(hist=hist, best_step=best_step, best_monitor_nll=best, train_secs=time.time() - t0)
        if best_state:
            model.load_state_dict(best_state)

    if mode != "timing":
        ppl, nll, n = L.perplexity(model, data.valid_blocks, device, cfg.get("eval_batch", 32), amp_dtype)
        res.update(val_ppl=ppl, val_nll=nll, n_tokens=n)
        log(f"validacion completa: perplejidad {ppl:.3f} (nll {nll:.4f}, {n} tokens)")
        if cfg.get("sampling"):
            res["sampling"] = sampling_grid(model, data, device, amp_dtype, log)
    res["secs"] = time.time() - t0
    if device.type == "cuda":
        res["max_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        res["max_reserved_gb"] = torch.cuda.max_memory_reserved() / 2 ** 30
    save_json(path, res)
    log(f"fin {rnd}/{run_id}: {res['secs'] / 60:.1f} min, memoria {res.get('max_mem_gb', 0):.2f} GB")
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_results(out_dir, rnd=None):
    pattern = os.path.join(out_dir, rnd or "*", "*", "result.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def expand(rnd, out_dir):
    """Devuelve las corridas de la ronda; las rondas de promocion toman las mejores de la ronda anterior."""
    spec = sweeps.ROUNDS[rnd]
    if "desde" not in spec:
        return spec
    ranked = sorted(load_results(out_dir, spec["desde"]), key=lambda r: r["val_ppl"])[:spec["top"]]
    return {f"{spec['prefijo']}_{r['id']}": dict(r["cfg"], **spec["cambios"]) for r in ranked}


def correr(exp_dir, data, sesion=None):
    out_dir = os.path.join(exp_dir, "runs")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_path = os.path.join(out_dir, "referencia.json")
    if not os.path.exists(ref_path):
        save_json(ref_path, dict(valid_techo_256=L.reference_validity(data.tok, data.valid, n=N_VALIDEZ, total_tokens=256)))
    progreso = Log(os.path.join(exp_dir, "progreso.txt"))
    for rnd in sesion or sweeps.SESION:
        runs = expand(rnd, out_dir)
        for run_id, cfg in runs.items():
            try:
                run_one(rnd, run_id, cfg, data, out_dir, device)
                progreso(f"{rnd}/{run_id} listo")
            except Exception as e:      # p. ej. falta de memoria: se registra y sigue con la siguiente
                progreso(f"{rnd}/{run_id} ERROR: {type(e).__name__} {str(e)[:300]}")
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    progreso("sesion terminada")


def resumen(exp_dir):
    rows = load_results(os.path.join(exp_dir, "runs"))
    table = []
    for r in rows:
        cfg = r["cfg"]
        row = dict(round=r["round"], id=r["id"], kind=cfg["kind"].split("/")[-1], params_M=round(r["n_params"] / 1e6, 1),
                   val_ppl=round(r["val_ppl"], 3) if "val_ppl" in r else "", min=round(r["secs"] / 60, 1),
                   mem_gb=round(r.get("max_mem_gb", 0), 2), min_epoca=round(r.get("min_per_epoch", 0), 1))
        if "sampling" in r:
            top = max(r["sampling"], key=lambda s: s["valid"])
            row["mejor_validez"] = round(top["valid"], 1)
        table.append(row)
    table.sort(key=lambda t: (t["round"], t["val_ppl"] if t["val_ppl"] != "" else 1e9))
    fields = ["round", "id", "kind", "params_M", "val_ppl", "mejor_validez", "min", "mem_gb", "min_epoca"]
    with open(os.path.join(exp_dir, "resumen.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(table)
    print(" | ".join(fields))
    for t in table:
        print(" | ".join(str(t.get(k, "")) for k in fields))

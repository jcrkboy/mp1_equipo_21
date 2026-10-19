"""Exporta un resumen compacto de todas las corridas (para embeber en el notebook y en el reporte).

Uso: python export_results.py <dir runs> <salida.csv>
"""
import glob
import json
import os
import sys

import pandas as pd

FAMILY = {"lstm": "LSTM", "gru": "GRU", "rnn": "RNN simple", "transformer": "Transformer causal",
          "tcn": "TCN causal", "mlp": "MLP ventana"}
HEAD = {"gmm": "Mezcla de Gaussianas", "gauss": "Gaussiana bivariada", "mse": "MSE (determinista)"}


CHANGE = {
    "cw_sqrt": "pesos de clase ∝ 1/√frecuencia en la CE",
    "cw_inv": "pesos de clase ∝ 1/frecuencia en la CE",
    "wpen3": "w_pen = 3",
    "mseaux1": "+ 1·MSE de la media de la mezcla",
    "combo": "MSE aux. 1 + w_pen 3 + pesos √",
    "mix10": "M = 10 componentes",
    "lr2e3": "lr = 2e-3",
    "bs64": "lote = 64",
    "d01": "dropout 0.1",
    "d03": "dropout 0.3",
    "aug": "aumento: escalado aleatorio ±15 %",
}


def change_of(name):
    for k, v in CHANGE.items():
        if name.endswith("_" + k):
            return v
    return ""


def describe(c):
    m = c["model"]
    if m in ("lstm", "gru", "rnn"):
        arch = f"{FAMILY[m]} {c['hidden']}" + (f"x{c['layers']}" if c["layers"] > 1 else "")
        if c.get("conv_front"):
            front = "TCN" if c.get("front_type") == "tcn" else f"{c.get('conv_layers', 2)}×Conv1d k{c.get('conv_kernel', 3)}"
            arch = f"{front} ({c['conv_front']}) + " + arch
        if c.get("abs_pos"):
            arch += " + posición integrada"
    elif m == "transformer":
        arch = f"Transformer d={c['d_model']} x{c['layers']}" + (f" (ventana {c['window']})" if c["window"] else "")
    elif m == "tcn":
        arch = f"TCN {c['channels']}c x{c['levels']} (campo receptivo {1 + 2 * (c['kernel'] - 1) * (2 ** c['levels'] - 1)})"
    else:
        arch = f"MLP {c['hidden']} (contexto c={c['context']})"
    return arch


def main(runs_dir, out_csv):
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "result.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        c, v = r["cfg"], r["val"]
        if r["name"].startswith("smoke"):
            continue
        rows.append({
            "ronda": r["name"].split("_")[0], "corrida": r["name"], "arquitectura": describe(c),
            "salida (Δx,Δy)": HEAD[c["head"]] + (f" M={c['n_mix']}" if c["head"] == "gmm" else ""),
            "params": r["n_params"], "épocas": r["epochs"], "mejor época": r["best_epoch"],
            "min": round(r["train_time"] / 60, 1),
            "val pérdida": round(v["loss"], 4), "val MSE": round(v["mse_norm"], 4),
            "val MSE px²": round(v["mse_px"], 1), "val F1 macro": round(v["f1_macro"], 4),
            "val F1 macro (umbral)": round(r["val_f1_tuned"], 4),
            "F1 fin trazo": round(v["f1_up"], 4), "F1 fin dibujo": round(v["f1_end"], 4),
            "gen puntos": round(r["gen"]["gen_len"], 1), "gen trazos": round(r["gen"]["gen_strokes"], 2),
            "cambio": change_of(r["name"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 50):
        print(df.drop(columns=["cambio"]).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

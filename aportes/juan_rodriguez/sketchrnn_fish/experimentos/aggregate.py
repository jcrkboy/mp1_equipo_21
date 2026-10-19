"""Junta todos los result.json de un directorio de resultados en results.csv y muestra un ranking."""
import glob
import json
import os
import sys

import pandas as pd

KEYS = ["model", "hidden", "layers", "dropout", "d_model", "window", "channels", "levels", "context",
        "head", "n_mix", "w_pen", "mse_aux", "class_weights", "label_smoothing", "augment",
        "batch_size", "lr", "weight_decay", "sched", "seed", "in_proj", "sigma_min"]


def collect(out_dir):
    rows = []
    for path in glob.glob(os.path.join(out_dir, "*", "result.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        v = r["val"]
        row = dict(name=r["name"])
        row.update({k: r["cfg"].get(k) for k in KEYS})
        row.update(params=r["n_params"], epochs=r["epochs"], best_ep=r["best_epoch"],
                   minutes=round(r["train_time"] / 60, 1),
                   val_loss=v["loss"], val_mse=v["mse_norm"], val_mse_px=v["mse_px"],
                   val_f1=v["f1_macro"], val_f1_tuned=r["val_f1_tuned"],
                   f1_down=v["f1_down"], f1_up=v["f1_up"], f1_end=v["f1_end"],
                   gen_len=r["gen"]["gen_len"], gen_strokes=r["gen"]["gen_strokes"],
                   gen_end_rate=r["gen"]["gen_end_rate"],
                   real_len=r["real"]["len"], real_strokes=r["real"]["strokes"])
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["score"] = df["val_f1"] - df["val_mse"]
        df = df.sort_values("score", ascending=False)
    return df


def main(out_dir):
    df = collect(out_dir)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    cols = ["name", "params", "best_ep", "minutes", "val_loss", "val_mse", "val_mse_px", "val_f1",
            "val_f1_tuned", "f1_up", "f1_end", "gen_len", "gen_strokes", "score"]
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 50):
        print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])

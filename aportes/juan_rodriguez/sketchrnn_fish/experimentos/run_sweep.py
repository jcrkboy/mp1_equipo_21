"""Ejecuta un barrido de configuraciones (definido en sweeps.py) en procesos paralelos.

Uso: python run_sweep.py --sweep r1 --data <dir .npz> --out <dir resultados> --workers 4 --threads 4
"""
import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _work(job):
    cfg, data_dir, out_dir = job
    import sketchlib
    try:
        return sketchlib.run_experiment(cfg, data_dir, out_dir)
    except Exception:
        return dict(name=cfg.get("name"), error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import sweeps
    cfgs = getattr(sweeps, args.sweep)()
    jobs = []
    for c in cfgs:
        c = dict(c, threads=args.threads)
        if not args.force and os.path.exists(os.path.join(args.out, c["name"], "result.json")):
            print("ya existe:", c["name"], flush=True)
            continue
        jobs.append((c, args.data, args.out))
    print(f"{len(jobs)} corridas, {args.workers} procesos x {args.threads} hilos", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_work, j): j[0]["name"] for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            if "error" in r:
                print("ERROR", r["name"], r["error"][-3000:], flush=True)
                continue
            v = r["val"]
            print(f"[{(time.time() - t0) / 60:6.1f} min] {r['name']:32s} loss={v['loss']:8.4f} "
                  f"mse={v['mse_norm']:.4f} f1={v['f1_macro']:.4f} f1t={r['val_f1_tuned']:.4f} "
                  f"len={r['gen']['gen_len']:.1f} strk={r['gen']['gen_strokes']:.2f} "
                  f"ep={r['best_epoch']}/{r['epochs']} t={r['train_time'] / 60:.1f}m", flush=True)
    import aggregate
    aggregate.main(args.out)


if __name__ == "__main__":
    main()

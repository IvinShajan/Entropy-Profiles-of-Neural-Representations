"""Training-dynamics experiment: how does layer rigidity evolve with epochs?

Trains a few configs while saving checkpoints every 2 epochs, then computes
the entropy-profile rigidity index at each checkpoint (plus acc/ECE).

Usage: python3 run_dynamics.py
"""
from __future__ import annotations
import os, json, sys, time
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
import analyze as az
import train_networks as tn
import metrics as mt

RES = os.path.join(BASE, "results")
NET_DIR = os.path.join(RES, "nets")


def analyse_checkpoint(label, cfg, epoch):
    net, _ = tn.build_model(cfg)
    sd = torch.load(os.path.join(NET_DIR, f"ckpt_{label}_ep{epoch}.pt"),
                    map_location="cpu")
    net.load_state_dict(sd)
    net.eval()
    xs, ys = az.get_probe(cfg["dataset"])
    reps = az.get_reps(net, xs)
    R = [az.layer_stats(A)[0]["R_pc1"] for A in reps]
    Rgram = [az.layer_stats(A)[0]["R_gram"] for A in reps]
    ms = mt.full_metric_suite(net, xs, ys)
    return dict(epoch=epoch, R_pc1=R, R_gram=Rgram, acc=ms["acc"], ece=ms["ece"])


def main():
    picks = [
        ("mnist", "mlp2", 12),
        ("mnist", "cnn2", 12),
        ("fashion", "cnn2", 12),
    ]
    for dataset, arch, epochs in picks:
        cfg = dict(dataset=dataset, arch=arch, epochs=epochs, dropout=0.0,
                   wd=0.0, lr=1e-3, init_scale=1.0, seed=99,
                   label=f"dyn-{dataset}-{arch}", max_train=None)
        print(f"=== training {cfg['label']} with checkpoints ===", flush=True)
        tn.train_one(cfg, save_checkpoints=True)
        rows = []
        for ep in range(2, epochs + 1, 2):
            rows.append(analyse_checkpoint(cfg["label"], cfg, ep))
            print(f"  ep{ep}: acc={rows[-1]['acc']:.3f} "
                  f"R={[round(r,3) for r in rows[-1]['R_pc1']]}", flush=True)
        # final epoch (not saved by the every-2 cadence if epochs odd; saved if even)
        with open(os.path.join(RES, f"dynamics_{cfg['label']}.json"), "w") as f:
            json.dump(dict(cfg=cfg, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()

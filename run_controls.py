"""Control experiments: analyse UNTRAINED (random-init) networks as the
'generic / random' baseline, and a Gaussian-mixture synthetic control.

This is the key control: untrained networks should have near-Poisson (R ~ 0)
entropy profiles, isolating what training does to rigidity.

Usage: python3 run_controls.py
"""
from __future__ import annotations
import os, json, sys
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
import analyze as az
import train_networks as tn
import entropy_profiles as ep

RES = os.path.join(BASE, "results")
os.makedirs(os.path.join(RES, "analysis"), exist_ok=True)


def gaussian_control():
    """Synthetic i.i.d. Gaussian feature batches -> PC1 config stats."""
    out = {}
    for ndim in [256, 1024]:
        rng = np.random.default_rng(ndim)
        A = rng.standard_normal((512, ndim))
        st, prof = az.layer_stats(A)
        out[f"gauss_{ndim}"] = st
    # Gaussian mixture with class structure (like real features)
    rng = np.random.default_rng(7)
    means = rng.standard_normal((10, 256)) * 2.0
    A = np.concatenate([rng.standard_normal((51, 256)) + means[c] for c in range(10)])
    st, prof = az.layer_stats(A)
    out["gauss_mixture"] = st
    return out


def main():
    controls = {}
    controls["synthetic"] = gaussian_control()

    # random-init real architectures on MNIST
    for arch in ["mlp2", "mlp4", "cnn2", "cnn3"]:
        cfg = dict(dataset="mnist", arch=arch, dropout=0.0, init_scale=1.0,
                   width=1, seed=0, label=f"random-{arch}")
        net, _ = tn.build_model(cfg)
        net.eval()
        xs, ys = az.get_probe("mnist")
        reps = az.get_reps(net, xs)
        layers = []
        for li, A in enumerate(reps):
            st, _ = az.layer_stats(A)
            st["layer_index"] = li
            layers.append(st)
        controls[f"random_{arch}"] = layers
        print(f"random-{arch}: R_pc1 = "
              f"{[round(l['R_pc1'],3) for l in layers]} "
              f"R_gram = {[round(l['R_gram'],3) for l in layers]}")

    with open(os.path.join(RES, "controls.json"), "w") as f:
        json.dump(controls, f, indent=1)
    print("controls saved to results/controls.json")


if __name__ == "__main__":
    main()

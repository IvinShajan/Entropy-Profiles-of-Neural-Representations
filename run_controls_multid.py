"""Multi-dimensional rigidity controls (Nature upgrade).

R_knn and the pairwise-distance entropy must behave like the 1D index:
untrained (random-init) networks and i.i.d. Gaussian clouds should score ~0
relative to the matched-radius Poisson reference, while a lattice should score
clearly positive. Clustered (over-dispersed) data should score ~0 or negative
(generic index), confirming the measure responds to order, not just shape.

Output: results/controls_multid.json
"""
import os, sys, json
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
RES = os.path.join(BASE, "results")

import train_networks as tn
import entropy_profiles as ep
from analyze import get_probe, get_reps

OUT = os.path.join(RES, "controls_multid.json")


def rep_stats(X):
    s = {}
    s["R_knn"] = ep.rigidity_index_knn_whitened(X, D_eff=12, seed=0)
    s["R_knn_generic"] = ep.knn_poisson_generic_index(X, D_eff=12, seed=0)
    s["pairwise_H"] = ep.pairwise_distance_entropy(X, sample_pairs=400_000)
    return s


def main():
    torch.set_num_threads(1)
    out = {"random_init": {}, "synthetic": {}, "lattice": {}}

    # random-init networks, per architecture
    for arch in ["mlp2", "mlp4", "cnn2", "cnn3"]:
        net, _ = tn.build_model(dict(dataset="mnist", arch=arch, dropout=0.0,
                                     init_scale=1.0, width=1, seed=0))
        net.eval()
        xs, _ = get_probe("mnist")
        reps = get_reps(net, xs)
        out["random_init"][arch] = [rep_stats(a) for a in reps]

    # synthetic clouds: i.i.d. Gaussian, 4-cluster mixture, lattice
    M, D = 510, 256
    rng = np.random.default_rng(0)
    gauss = rng.standard_normal((M, D))
    out["synthetic"]["iid_gauss"] = rep_stats(gauss)
    centers = np.zeros((4, D))
    centers[:, :4] = 6.0 * np.array([[1, 0, 0, 0], [-1, 0, 0, 0],
                                     [0, 1, 0, 0], [0, -1, 0, 0]])
    ci = rng.integers(0, 4, M)
    clust = centers[ci] + 0.1 * rng.standard_normal((M, D))
    out["synthetic"]["cluster_mix"] = rep_stats(clust)

    # D-dim lattice: 3D cubic grid embedded in the first 3 coordinates of the
    # D-dimensional space (numpy cannot materialise a 256-dim meshgrid)
    side = 8                                   # 8^3 = 512 ~ M
    coords = [np.arange(side)] * 3
    grid3 = np.stack(np.meshgrid(*coords, indexing="ij"), -1).reshape(-1, 3)
    grid = np.zeros((grid3.shape[0], D))
    grid[:, :3] = grid3
    grid = grid[:M].astype(float)
    out["lattice"]["cube3d"] = rep_stats(grid)

    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)
    print("random-init R_knn by arch/layer:")
    for arch, ls in out["random_init"].items():
        print(f"  {arch}: {[round(l['R_knn'], 3) for l in ls]}")
    print("iid_gauss R_knn=%.3f (gen=%.3f) H=%.3f"
          % (out["synthetic"]["iid_gauss"]["R_knn"],
             out["synthetic"]["iid_gauss"]["R_knn_generic"],
             out["synthetic"]["iid_gauss"]["pairwise_H"]))
    print("cluster  R_knn=%.3f (gen=%.3f)"
          % (out["synthetic"]["cluster_mix"]["R_knn"],
             out["synthetic"]["cluster_mix"]["R_knn_generic"]))
    print("lattice  R_knn=%.3f (gen=%.3f)"
          % (out["lattice"]["cube3d"]["R_knn"],
             out["lattice"]["cube3d"]["R_knn_generic"]))


if __name__ == "__main__":
    main()

"""Analysis pipeline: extract per-layer representations and compute
entropy-profile statistics (rigidity index, gap stats, entropy profiles),
plus CKA / effective-rank baselines.

Usage:
    python3 analyze.py            # analyse every trained network in results/nets
    python3 analyze.py --label X  # analyse one network
"""
from __future__ import annotations
import os, json, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
import entropy_profiles as ep
import metrics as mt
import train_networks as tn

RES = os.path.join(BASE, "results")
NET_DIR = os.path.join(RES, "nets")
PROBE_DIR = os.path.join(RES, "probes")
os.makedirs(PROBE_DIR, exist_ok=True)
torch.set_num_threads(2)


# ---------------------------------------------------------------------------
# Probe sets (balanced subset of the test split)
# ---------------------------------------------------------------------------

def probe_for(dataset, n=512, seed=123):
    """Balanced (n//n_classes per class) probe set of images + labels."""
    dtest = tn.load_dataset(dataset, train=False)
    nc = tn.n_classes_of(dataset)
    per = n // nc
    idx = []
    for c in range(nc):
        ci = [i for i, (_, y) in enumerate(dtest) if int(y) == c]
        rng = np.random.default_rng(seed)
        idx.extend(list(rng.choice(ci, min(per, len(ci)), replace=False)))
    xs = torch.stack([torch.as_tensor(dtest[i][0]) for i in idx])
    ys = torch.tensor([int(dtest[i][1]) for i in idx])
    return xs, ys


def get_probe(dataset):
    path = os.path.join(PROBE_DIR, f"probe_{dataset}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return torch.from_numpy(d["x"]), torch.from_numpy(d["y"])
    x, y = probe_for(dataset)
    np.savez(path, x=x.numpy(), y=y.numpy())
    return x, y


# ---------------------------------------------------------------------------
# Representation extraction
# ---------------------------------------------------------------------------

def get_reps(net, x):
    """Activations of the semantic feature layers, in order, as numpy arrays.

    For MLP/CNN models: conv blocks (Sequential containing Conv2d) and
    post-activation hidden MLP layers (ReLU children). For torchvision ResNets:
    the 4 stage blocks (Sequential modules containing Bottleneck/BasicBlock)
    followed by the post-pool 512-dim feature layer (the 'fc.in_features' view).
    Logits excluded.
    """
    captured = {}
    hooked = []
    handles = []

    # Detect ResNet-style models: top-level children include nn.Sequential
    # whose submodules contain torchvision BasicBlock/Bottleneck, plus a 'fc'.
    is_resnet = hasattr(net, "fc") and any(
        isinstance(m, nn.Sequential)
        and any(type(m2).__name__ in ("BasicBlock", "Bottleneck")
                for m2 in m.modules())
        for m in net.children())
    if is_resnet:
        for name, child in net.named_children():
            if name == "fc":
                continue
            def make_resnet(name_):
                def h(m, inp, out):
                    a = out.detach().float().cpu().numpy()
                    captured[name_] = a
                return h
            handles.append(child.register_forward_hook(make_resnet(name)))
            hooked.append(name)
        # also capture the avgpool output (penultimate 512-dim representation)
        if hasattr(net, "avgpool"):
            def make_pool(name_="avgpool"):
                def h(m, inp, out):
                    captured[name_] = out.detach().float().cpu().numpy()
                return h
            handles.append(net.avgpool.register_forward_hook(make_pool()))
            hooked.append("avgpool")
        with torch.no_grad():
            net(x)
        for h in handles:
            h.remove()
        reps = []
        for name in hooked:
            a = captured[name]
            if a.ndim == 4:
                b, c, hh, ww = a.shape
                a = a.reshape(b, c * hh * ww)
            reps.append(a)
        return reps

    for idx, child in enumerate(net.children()):
        is_conv_block = (isinstance(child, nn.Sequential)
                         and any(isinstance(m, nn.Conv2d) for m in child.modules()))
        is_gap = (isinstance(child, nn.Sequential)
                  and any(isinstance(m, nn.AdaptiveAvgPool2d) for m in child.modules()))
        if isinstance(child, nn.ReLU) or is_conv_block or is_gap:
            def make(i):
                def h(m, inp, out):
                    captured[i] = out.detach().float().cpu().numpy()
                return h
            handles.append(child.register_forward_hook(make(idx)))
            hooked.append(idx)
    with torch.no_grad():
        net(x)
    for h in handles:
        h.remove()
    reps = []
    for i in hooked:
        a = captured[i]
        if a.ndim == 4:
            b, c, hh, ww = a.shape
            a = a.reshape(b, c * hh * ww)
        reps.append(a)
    return reps


# ---------------------------------------------------------------------------
# Point-configuration statistics per layer
# ---------------------------------------------------------------------------

def pc1_projection(A):
    Ac = A - A.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Ac, full_matrices=False)
    return Ac @ Vt[0]


def gram_eigs(A):
    d = max(A.shape[1], 1)
    G = (A @ A.T) / d
    return np.linalg.eigvalsh(G)


def vector_coord_rigidity(A, nsample=120, max_coords=1024, seed=0):
    """Rigidity of the coordinate values of a single input's activation vector,
    averaged over inputs (the 'single input across features' view). Large
    feature spaces are subsampled to max_coords coordinates."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(A), min(nsample, len(A)), replace=False)
    Rs = []
    for i in idx:
        v = A[i]
        if v.size > max_coords:
            v = rng.choice(v, max_coords, replace=False)
        cfg = ep.unfold_kde(np.sort(v))
        try:
            Rs.append(ep.rigidity_index(cfg, seed=seed))
        except ValueError:
            continue
    return float(np.mean(Rs)) if Rs else float("nan")


def linear_cka(X, Y):
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    K = Xc @ Xc.T
    L = Yc @ Yc.T
    hsic = float(np.sum(K * L))
    den = float(np.sqrt(np.sum(K * K) * np.sum(L * L)))
    return hsic / den if den > 0 else float("nan")


def participation_ratio(A):
    s = np.linalg.svd(A, compute_uv=False)
    s = s / (s.sum() + 1e-12)
    return float((s.sum()) ** 2 / ((s ** 2).sum() + 1e-12))


def layer_stats(A, nL=8, seed=0):
    """All entropy-profile statistics for one layer's activation matrix."""
    st = {}
    # --- PC1 point configuration
    pc = pc1_projection(A)
    cfg_pc = ep.unfold_kde(pc)
    R_pc, Hx, Hp, Hl, Ls = ep.rigidity_index(cfg_pc, nLs=nL, seed=seed, return_parts=True)
    g_pc = ep.normalize_gaps(cfg_pc)
    kp = ep.ks_poisson(g_pc)
    kg = ep.ks_gue(g_pc)
    st["R_pc1"] = R_pc
    st["gap_var_pc1"] = ep.gap_var(g_pc)
    st["ks_pois_pc1"] = float(kp[0])
    st["p_pois_pc1"] = float(kp[1])
    st["ks_gue_pc1"] = float(kg[0])
    st["p_gue_pc1"] = float(kg[1])
    st["llr_pc1"] = ep.log_likelihood_ratio(g_pc)
    st["slope_pc1"] = ep.profile_slope(cfg_pc)
    st["nvar20_pc1"] = ep.number_variance(cfg_pc, 20.0)
    st["delta3_pc1"] = ep.delta3(cfg_pc, 20.0)
    # --- Gram eigenvalue configuration
    cfg_gr = ep.unfold_kde(gram_eigs(A))
    st["R_gram"] = ep.rigidity_index(cfg_gr, nLs=nL, seed=seed)
    st["llr_gram"] = ep.log_likelihood_ratio(ep.normalize_gaps(cfg_gr))
    # --- single-input coordinate view
    st["R_vec"] = vector_coord_rigidity(A, seed=seed)
    # --- multi-dimensional (whitened k-NN) views  [Nature upgrade]
    try:
        st["R_knn"] = ep.rigidity_index_knn_whitened(A, D_eff=12, seed=seed)
    except Exception:
        st["R_knn"] = float("nan")
    try:
        st["R_knn_generic"] = ep.knn_poisson_generic_index(A, D_eff=12, seed=seed)
    except Exception:
        st["R_knn_generic"] = float("nan")
    try:
        st["pairwise_H"] = ep.pairwise_distance_entropy(A, sample_pairs=400_000)
    except Exception:
        st["pairwise_H"] = float("nan")
    # --- baselines
    st["eff_rank"] = participation_ratio(A)
    st["npoints"] = int(A.shape[0])
    st["ndim"] = int(A.shape[1])
    return st, dict(Ls=Ls, Hx=Hx, Hp=Hp, Hl=Hl, g=g_pc, cfg=cfg_pc)


# ---------------------------------------------------------------------------
# Full per-network analysis
# ---------------------------------------------------------------------------

def analyse_network(label, cfg):
    net, _ = tn.build_model(cfg)
    sd = torch.load(os.path.join(NET_DIR, f"model_{label}.pt"), map_location="cpu")
    net.load_state_dict(sd)
    net.eval()
    xs, ys = get_probe(cfg["dataset"])
    reps = get_reps(net, xs)

    out = {"label": label, "cfg": cfg}
    # headline metrics on the same probe set
    ms = mt.full_metric_suite(net, xs, ys)
    out["metrics"] = ms
    # generalization gap from training log
    try:
        log = json.load(open(os.path.join(NET_DIR, f"log_{label}.json")))
        out["train_acc_final"] = log["train_acc"][-1]
        out["test_acc_final"] = log["test_acc"][-1]
        out["gen_gap"] = log["train_acc"][-1] - log["test_acc"][-1]
    except Exception:
        pass

    # CKA between consecutive layers and input->layer1
    cka_chain = []
    prev = reps[0]
    for i, a in enumerate(reps[1:], start=1):
        cka_chain.append(linear_cka(prev, a))
        prev = a
    out["cka_chain"] = cka_chain
    out["R_input_layer"] = linear_cka(xs.numpy().reshape(len(xs), -1), reps[0])

    layers = []
    profiles = {"Ls": None}
    for li, A in enumerate(reps):
        st, prof = layer_stats(A)
        st["layer_index"] = li
        layers.append(st)
        if profiles["Ls"] is None:
            profiles["Ls"] = prof["Ls"].tolist()
        profiles[f"Hx_l{li}"] = prof["Hx"].tolist()
        profiles[f"Hp_l{li}"] = prof["Hp"].tolist()
        profiles[f"Hl_l{li}"] = prof["Hl"].tolist()
    out["layers"] = layers
    return out, profiles


def main(args):
    cfgs = tn.configs()
    os.makedirs(os.path.join(RES, "profiles"), exist_ok=True)
    for c in cfgs:
        if args.label and c["label"] != args.label:
            continue
        label = c["label"]
        out_path = os.path.join(RES, "analysis", f"{label}.json")
        os.makedirs(os.path.join(RES, "analysis"), exist_ok=True)
        model_path = os.path.join(NET_DIR, f"model_{label}.pt")
        if not os.path.exists(model_path):
            continue
        if os.path.exists(out_path) and not args.force:
            continue
        t0 = time.time()
        try:
            out, profiles = analyse_network(label, c)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=1)
            np.savez(os.path.join(RES, "profiles", f"{label}.npz"), **profiles)
            print(f"OK {label} ({time.time()-t0:.0f}s) "
                  f"acc={out['metrics']['acc']:.3f} "
                  f"R_layers={[round(l['R_pc1'],2) for l in out['layers']]}")
        except Exception as e:
            import traceback
            print(f"FAIL {label}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(args)

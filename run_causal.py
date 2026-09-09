"""Causal rigidity experiment (Nature upgrade).

Correlational results (rigidity -> generalization gap) leave the causal
direction open: does *imposing* level-statistics rigidity in the hidden
activations during training reduce overfitting, or does rigidity merely
accompany well-behaved training?

Design: small MNIST models are trained in an overfitting regime (small train
subset) under a differentiable penalty that pushes the eigenvalue spacing of
each hidden layer's Gram matrix toward *regular* (lattice/GUE-like) order.
The penalty is the coefficient of variation of the log-normalized eigenvalue
gaps of the top-k Gram spectrum -- smooth, and zero for perfectly regular
spacings (maximal rigidity), ~1 for Poisson spectra (generic).

Arms:
  control            : Cross-entropy only (lambda = 0)
  penultimate        : rigidity penalty on the last hidden layer only
  all-hidden         : rigidity penalty on every hidden layer
  spread             : control -- spectral-concentration penalty (same
                       penalty family, same lambda scale) that shrinks the
                       eigenvalue distribution WITHOUT enforcing regular
                       spacings. Isolates "spacing order" from "spectral
                       scale" as the causal ingredient.
  spread-rigid       : rigidity penalty (as penultimate) named for pairing
                       with the spread arm in a same-lambda comparison.

For each arm we record test accuracy, generalization gap, and the *measured*
1D rigidity index R_avg of the final penultimate activations (verify the
penalty moved rigidity, and whether that moved the gap).
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import train_networks as tn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_DIR = os.path.join(BASE, "results", "causal")
os.makedirs(NET_DIR, exist_ok=True)


def gap_cv_penalty(A, k=24, eps=1e-6):
    """Spacing-regularity penalty on the Gram spectrum of a batch activation.

    A : (B, D) batch activations. Returns a scalar, differentiable penalty:
      eigenvalues of G = (1/B) A A^T + eps I, take the top k, form gaps,
      normalize by their mean, and return var(log gap) (0 = regular/lattice
      spacings; ~1 = Poisson-like). Minimizing drives activations rigid.
    """
    B = A.shape[0]
    G = (A @ A.t()) / B
    eig = torch.linalg.eigvalsh(G)
    eig = eig[-(k + 1):]                       # keep top k+1 eigenvalues
    gaps = torch.diff(eig)                     # k gaps
    gaps = torch.clamp(gaps, min=eps)
    gaps = gaps / gaps.mean().detach() + eps
    return torch.var(torch.log(gaps))


def spread_penalty(A, k=24, eps=1e-6):
    """Control penalty: collapse the spectral SCALE without spacing order.

    Minimizing this shrinks the coefficient of variation of the eigenvalues
    themselves (concentrating the spectrum, as plain ridge/decay would) but
    does NOT force the *gaps* to be regular. Matching this arm against the
    rigidity arm at equal lambda separates "local spacing order" from
    "spectral concentration" as the causal ingredient.
    """
    B = A.shape[0]
    G = (A @ A.t()) / B
    eig = torch.linalg.eigvalsh(G)
    eig = eig[-(k + 1):] + eps
    eig = eig / eig.mean().detach() + eps
    return torch.var(torch.log(eig))


class PenaltyNet(nn.Module):
    """Wraps a model and collects hidden activations per forward pass."""

    def __init__(self, net, hidden_names):
        super().__init__()
        self.net = net
        self.hidden_names = hidden_names
        self.activations = {}
        self.handles = []
        mods = dict(net.named_modules())
        for name in hidden_names:
            m = mods[name]
            self.handles.append(m.register_forward_hook(self._mk(name)))

    def _mk(self, name):
        def h(m, inp, out):
            a = out
            if isinstance(a, (list, tuple)):
                a = a[0]
            if a.ndim > 2:
                a = F.adaptive_avg_pool2d(a, 4).flatten(1)
            self.activations[name] = a      # keep graph for the penalty
        return h

    def forward(self, x):
        self.activations = {}
        return self.net(x)

    def remove(self):
        for h in self.handles:
            h.remove()


def build_with_hooks(arch):
    if arch == "mlp2":
        net, dims = tn.mlp([784, 128, 10])
        hidden_names = ["1"]                       # ReLU after first Linear
    elif arch == "cnn2":
        net, dims = tn.cnn2()
        # ReLU outputs: conv-block 0 (idx "0.2"), conv-block 1 (idx "2.2"),
        # and the fully-connected ReLU (idx "7")
        hidden_names = ["0.2", "2.2", "7"]
    else:
        raise ValueError(arch)
    return PenaltyNet(net, hidden_names)


def train_arm(arch, arm, lam, n_train, epochs, seed, bs=128, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build_with_hooks(arch)

    ds = tn.load_dataset("mnist", train=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), n_train, replace=False)
    dtr = DataLoader(Subset(ds, idx), batch_size=bs, shuffle=True)
    dte = DataLoader(tn.load_dataset("mnist", train=False), batch_size=512)

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)
    lossf = nn.CrossEntropyLoss()

    def acc(loader):
        c = t = 0
        with torch.no_grad():
            for x, y in loader:
                c += (net(x).argmax(1) == y).sum().item()
                t += y.size(0)
        return c / t

    log = {"train_acc": [], "test_acc": []}
    for ep in range(epochs):
        net.train()
        for x, y in dtr:
            opt.zero_grad()
            logits = net(x)
            loss = lossf(logits, y)
            if arm in ("penultimate", "all-hidden", "spread-rigid"):
                pen = gap_cv_penalty
            elif arm == "spread":
                pen = spread_penalty
            else:
                pen = None
            if arm != "control" and lam > 0 and pen is not None:
                for name, a in net.activations.items():
                    if arm in ("penultimate", "spread-rigid", "spread") and \
                       name != net.hidden_names[-1]:
                        continue
                    loss = loss + lam * pen(a)
            loss.backward()
            opt.step()
        net.eval()
        if ep == epochs - 1:                     # eval only at the end
            log["train_acc"].append(acc(dtr))
            log["test_acc"].append(acc(dte))

    # measured rigidity of the penultimate hidden layer after training
    net.eval()
    from analyze import probe_for, get_reps, layer_stats
    xp, yp = probe_for("mnist", n=512, seed=seed)
    reps = get_reps(net.net, xp)
    meas = {}
    for i, a in enumerate(reps):
        st, _ = layer_stats(a)
        meas[f"R_avg_l{i}"] = st["R_pc1"]
    meas["R_avg_penult"] = meas[f"R_avg_l{len(reps) - 1}"]
    meas["R_avg_mean"] = float(np.mean([v for k, v in meas.items()
                                        if k.startswith("R_avg_l")]))
    meas["R_knn_penult"] = layer_stats(reps[-1])[0]["R_knn"]

    return {
        "arch": arch, "arm": arm, "lam": lam, "seed": seed,
        "n_train": n_train, "epochs": epochs,
        "train_acc_final": log["train_acc"][-1],
        "test_acc_final": log["test_acc"][-1],
        "gen_gap": log["train_acc"][-1] - log["test_acc"][-1],
        "train_acc": log["train_acc"], "test_acc": log["test_acc"],
        "measured": meas,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mlp2", choices=["mlp2", "cnn2"])
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lambdas", type=str, default="0,0.01,0.1,0.5,1.0")
    ap.add_argument("--arms", type=str,
                    default="control,penultimate,all-hidden,spread")
    args = ap.parse_args()

    lams = [float(v) for v in args.lambdas.split(",")]
    arms = args.arms.split(",")
    out = {"args": vars(args), "runs": []}
    for arm in arms:
        for lam in lams:
            if arm == "control" and lam > 0:
                continue
            for s in range(args.seeds):
                t0 = time.time()
                r = train_arm(args.arch, arm, lam, args.n_train,
                              args.epochs, s)
                r["seconds"] = time.time() - t0
                out["runs"].append(r)
                print(f"[{arm} lam={lam} s={s}] test={r['test_acc_final']:.3f} "
                      f"gap={r['gen_gap']:.3f} R_avg={r['measured']['R_avg_penult']:.3f} "
                      f"({r['seconds']:.0f}s)", flush=True)
    tag = f"{args.arch}_n{args.n_train}_e{args.epochs}"
    fp = os.path.join(NET_DIR, f"causal_{tag}.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", fp)


if __name__ == "__main__":
    main()

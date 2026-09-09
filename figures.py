"""All paper figures.

Usage: python3 figures.py
Produces PNGs in ../figures.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "code"))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)
import entropy_profiles as ep

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
C_POIS, C_GUE, C_LAT = "#1f77b4", "#d62728", "#2ca02c"
C_TRAIN, C_INIT = "#1f77b4", "#ff7f0e"


def save(fig, name: str) -> None:
    """Write the figure as both PNG (verification) and SVG (paper/docx)."""
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=150)
    fig.savefig(os.path.join(FIG, os.path.splitext(name)[0] + ".svg"))


# ---------------------------------------------------------------------------
# Fig 1: reference fingerprints (spacing densities + entropy profiles)
# ---------------------------------------------------------------------------
def fig1():
    M = 2000
    xs = {"Poisson": ep.poisson_points(M, seed=1),
          "GUE": ep.gue_points(M, seed=1),
          "Lattice": ep.lattice_points(M)}
    Ls = np.logspace(np.log10(2), np.log10(64), 12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # spacing densities
    ax = axes[0]
    for name, x in xs.items():
        g = ep.normalize_gaps(x)
        ax.hist(g, bins=60, density=True, alpha=0.45, label=name)
    s = np.linspace(0, 4, 300)
    ax.plot(s, np.exp(-s), "b--", lw=2, label=r"Poisson $e^{-s}$")
    ax.plot(s, (np.pi * s / 2) * np.exp(-np.pi * s ** 2 / 4), "r--", lw=2,
            label=r"GUE $\frac{\pi s}{2}e^{-\pi s^2/4}$")
    ax.set_xlabel("normalized spacing s")
    ax.set_ylabel("density")
    ax.set_title("(a) Nearest-neighbour spacing distributions")
    ax.legend(fontsize=8)

    # entropy profiles
    ax = axes[1]
    for name, x in xs.items():
        _, H = ep.entropy_profile(x, Ls)
        ax.plot(Ls, H, "o-", lw=2, label=name)
    ax.plot(Ls, 0.5 * np.log(2 * np.pi * np.e * Ls), "k:", lw=1.5,
            label=r"$\frac{1}{2}\ln(2\pi e L)$")
    ax.set_xscale("log")
    ax.set_xlabel("window length L (mean-spacing units)")
    ax.set_ylabel(r"window-count entropy $H(L)$ (nats)")
    ax.set_title("(b) Entropy profiles of the three ensembles")
    ax.legend(fontsize=8)

    # number variance
    ax = axes[2]
    for name, x in xs.items():
        _, V = ep.number_variance_profile(x, Ls)
        ax.plot(Ls, V, "o-", lw=2, label=name)
    ax.plot(Ls, Ls, "b:", lw=1.2, label="L (Poisson)")
    ax.plot(Ls, (2 / np.pi ** 2) * np.log(Ls), "r:", lw=1.2,
            label=r"$\frac{2}{\pi^2}\ln L$ (GUE)")
    ax.set_xscale("log")
    ax.set_xlabel("window length L")
    ax.set_ylabel(r"number variance $\Sigma^2(L)$")
    ax.set_title("(c) Number variance (spectral rigidity)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "fig1_references.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: rigidity index across layers — untrained vs trained
# (Honest null result: individual-layer rigidity does NOT separate the two.)
# ---------------------------------------------------------------------------
def fig2():
    controls = json.load(open(os.path.join(RES, "controls.json")))
    df = _load_df()
    init = {"R_pc1": [], "R_gram": [], "gapvar": []}
    for k, layers in controls.items():
        if k.startswith("random_"):
            for l in layers:
                init["R_pc1"].append(l["R_pc1"])
                init["R_gram"].append(l.get("R_gram", 0.0))
                init["gapvar"].append(l.get("gap_var_pc1", np.nan))
    trained = {"R_pc1": [], "R_gram": [], "gapvar": []}
    for _, r in df.iterrows():
        layers = r["layer_Rs"]
        trained["R_pc1"].extend(layers)
    # reload R_gram / gapvar per layer from analysis jsons
    import glob
    for f in glob.glob(os.path.join(RES, "analysis", "*.json")):
        d = json.load(open(f))
        for l in d["layers"]:
            trained["R_gram"].append(l["R_gram"])
            trained["gapvar"].append(l["gap_var_pc1"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    titles = ["(a) 1D spacing rigidity R",
              "(b) Gram-spectrum rigidity R",
              "(c) Spacing variance (Poisson = 1)"]
    keys = ["R_pc1", "R_gram", "gapvar"]
    for ax, key, ttl in zip(axes, keys, titles):
        u = [v for v in init[key] if np.isfinite(v)]
        tr = [v for v in trained[key] if np.isfinite(v)]
        bp = ax.boxplot([u, tr], tick_labels=["untrained\n(random init)", "trained"],
                        widths=0.5, patch_artist=True,
                        boxprops=dict(alpha=0.6))
        for j, arr in enumerate([u, tr], start=1):
            ax.scatter(np.random.default_rng(j).normal(j, 0.05, len(arr)),
                       arr, s=8, alpha=0.45)
        ax.set_ylabel(key)
        ax.set_title(ttl)
        if key == "gapvar":
            ax.axhline(1.0, color=C_POIS, ls=":", lw=1, label="Poisson (generic)")
            ax.axhline(0.19, color=C_GUE, ls="--", lw=1, label="GUE")
            ax.legend(fontsize=8)
    fig.suptitle("Individual-layer rigidity does not separate trained from "
                 "untrained networks", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_untrained_vs_trained.png")
    plt.close(fig)


def _load_df():
    import correlations as cr
    return cr.load_rows()


# ---------------------------------------------------------------------------
# Fig 3: representative entropy profiles of real layers vs references
# ---------------------------------------------------------------------------
def fig3():
    df = _load_df()
    Ls = np.logspace(np.log10(2), np.log10(64), 10)
    Hp = np.array([ep.window_entropy(ep.poisson_points(1200, seed=s), L)
                   for s in range(6) for L in Ls]).reshape(6, -1).mean(0)
    Hg = ep.entropy_profile(ep.gue_points(1200, seed=1), Ls)[1]
    Hl = ep.entropy_profile(ep.lattice_points(1200), Ls)[1]

    import analyze as az, train_networks as tn

    def profile_of(label, net, xs, layer_idx):
        reps = az.get_reps(net, xs)
        A = reps[layer_idx]
        cfg_x = ep.unfold_kde(az.pc1_projection(A))
        _, Hx = ep.entropy_profile(cfg_x, Ls)
        return Hx

    xs, _ = az.get_probe("mnist")
    # random init network (generic control)
    net_r, _ = tn.build_model(dict(dataset="mnist", arch="mlp2", dropout=0.0,
                                   init_scale=1.0, width=1, seed=0))
    net_r.eval()
    # trained networks
    net_t, _ = tn.build_model(dict(dataset="mnist", arch="mlp2", dropout=0.0,
                                   init_scale=1.0, width=1, seed=0))
    net_t.load_state_dict(torch.load(os.path.join(RES, "nets", "model_mnist-mlp2-base.pt"),
                                     map_location="cpu"))
    net_t.eval()
    net_c, _ = tn.build_model(dict(dataset="mnist", arch="cnn2", dropout=0.0,
                                   init_scale=1.0, width=1, seed=0))
    net_c.load_state_dict(torch.load(os.path.join(RES, "nets", "model_mnist-cnn2-base.pt"),
                                     map_location="cpu"))
    net_c.eval()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax in axes:
        ax.plot(Ls, Hp, color=C_POIS, lw=2, label="Poisson (generic)")
        ax.plot(Ls, Hg, color=C_GUE, lw=2, label="GUE (intermediate)")
        ax.plot(Ls, Hl, color=C_LAT, lw=2, label="Lattice (rigid)")
        ax.set_xscale("log"); ax.set_xlabel("window length L (mean spacing)")
        ax.set_ylabel(r"window-count entropy $H(L)$ (nats)")
        ax.legend(fontsize=8)

    Hx = profile_of(None, net_r, xs, 0)
    axes[0].plot(Ls, Hx, "o--", color=C_INIT, ms=4, lw=1.5, label="random-init MLP (layer 1)")
    axes[0].set_title("(a) generic (untrained) representation")

    Hx = profile_of(None, net_t, xs, 0)
    axes[1].plot(Ls, Hx, "o--", color=C_TRAIN, ms=4, lw=1.5, label="trained MLP (hidden layer)")
    axes[1].set_title("(b) trained MLP")

    for li in [0, 2]:
        Hx = profile_of(None, net_c, xs, li)
        axes[2].plot(Ls, Hx, "o--", ms=4, lw=1.5,
                     label=f"trained CNN (conv layer {li + 1})")
    axes[2].set_title("(c) trained CNN")

    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig3_profiles.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4-6: rigidity vs behaviour scatter plots
# ---------------------------------------------------------------------------
def scatter(target, ylabel, fname, use="R_avg"):
    df = _load_df()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for ds, col, mk in [("mnist", C_TRAIN, "o"), ("fashion", C_POIS, "s"),
                        ("cifar-small", C_LAT, "^")]:
        m = df.dataset == ds
        x, y = df[use][m], df[target][m]
        ax.scatter(x, y, c=col, marker=mk, s=45, alpha=0.8, label=ds)
    m = np.isfinite(df[use]) & np.isfinite(df[target])
    rho, p = spearmanr(df[use][m], df[target][m])
    ax.set_xlabel(use.replace("_avg", "").replace("R_", "Rigidity R (") + ")" if use.startswith("R_") else use)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}\nSpearman rho={rho:.3f} (p={p:.2e})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, fname)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 7: training dynamics
# ---------------------------------------------------------------------------
def fig7():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    pairs = [("dyn-mnist-mlp2", "MLP2 / MNIST"),
             ("dyn-mnist-cnn2", "CNN2 / MNIST"),
             ("dyn-fashion-cnn2", "CNN2 / Fashion-MNIST")]
    for ax, (base, title) in zip(axes, pairs):
        fn = os.path.join(RES, f"dynamics_{base}.json")
        if not os.path.exists(fn):
            ax.text(0.5, 0.5, "no dynamics data", ha="center")
            continue
        d = json.load(open(fn))
        rows = d["rows"]
        ep_ = [r["epoch"] for r in rows]
        acc = [r["acc"] for r in rows]
        R = [np.mean(r["R_pc1"]) for r in rows]
        Rg = [np.mean(r["R_gram"]) for r in rows]
        ax.plot(ep_, R, "o-", color=C_TRAIN, label="R (mean layer)")
        ax.plot(ep_, Rg, "s-", color="#9467bd", label="R (Gram)")
        ax2 = ax.twinx()
        ax2.plot(ep_, acc, "s--", color=C_INIT, label="test acc")
        ax2.set_ylabel("test accuracy")
        ax.set_xlabel("epoch")
        ax.set_ylabel("Rigidity index R")
        ax.set_title(f"training dynamics ({title})")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig7_dynamics.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 8: predictive power comparison (rigidity vs baselines)
# ---------------------------------------------------------------------------
def fig8():
    df = _load_df()
    targets = {"gen_gap": "Gen. gap", "ece": "ECE", "fgsm03": "FGSM", "pgd": "PGD"}
    preds = {"R_avg": "Rigidity R", "R_gram_avg": "R (Gram)",
             "cka_avg": "Linear CKA", "cka_io": "CKA input-last",
             "effrank_avg": "Effective rank", "R_vec_avg": "R (coords)"}
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    xpos = np.arange(len(targets))
    width = 0.12
    for j, (pk, pl) in enumerate(preds.items()):
        rhos = []
        for tk in targets:
            m = np.isfinite(df[pk]) & np.isfinite(df[tk])
            rhos.append(abs(spearmanr(df[pk][m], df[tk][m])[0]))
        ax.bar(xpos + (j - len(preds) / 2) * width, rhos, width, label=pl)
    ax.set_xticks(xpos)
    ax.set_xticklabels(list(targets.values()))
    ax.set_ylabel("|Spearman rho|")
    ax.set_title("Predictive power of rigidity vs moment-based baselines")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save(fig, "fig8_predictive.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 9: multi-dimensional kNN rigidity (Nature upgrade)
# ---------------------------------------------------------------------------
def fig9():
    df = _load_df()
    import correlations as cr
    def rho(a, b):
        m = np.isfinite(df[a]) & np.isfinite(df[b])
        return spearmanr(df[a][m], df[b][m])[0], (df[a][m].values, df[b][m].values)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ds_style = [("mnist", C_TRAIN, "o"), ("fashion", C_POIS, "s"),
                ("cifar-small", C_LAT, "^")]

    # (a) R_knn vs accuracy (the measure is strongly accuracy-aligned)
    ax = axes[0, 0]
    for ds, col, mk in ds_style:
        m = df.dataset == ds
        ax.scatter(df.R_knn_avg[m], df.acc[m], c=col, marker=mk, s=45,
                   alpha=0.8, label=ds)
    r, _ = rho("R_knn_avg", "acc")
    ax.set_xlabel(r"multi-D rigidity $R_{\rm knn}$")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"(a) $R_{{knn}}$ vs accuracy  (rho={r:.2f})")
    ax.legend(fontsize=8)

    # (b) R_knn vs ECE with honest partial annotations
    ax = axes[0, 1]
    for ds, col, mk in ds_style:
        m = df.dataset == ds
        ax.scatter(df.R_knn_avg[m], df.ece[m], c=col, marker=mk, s=45,
                   alpha=0.8, label=ds)
    r, _ = rho("R_knn_avg", "ece")
    ax.set_xlabel(r"multi-D rigidity $R_{\rm knn}$")
    ax.set_ylabel("ECE")
    ax.set_title(f"(b) $R_{{knn}}$ vs calibration  (rho={r:.2f})")
    ax.legend(fontsize=8)

    # (c) R_knn vs gen gap (the honest null: dies after controlling accuracy)
    ax = axes[0, 2]
    for ds, col, mk in ds_style:
        m = df.dataset == ds
        ax.scatter(df.R_knn_avg[m], df.gen_gap[m], c=col, marker=mk, s=45,
                   alpha=0.8, label=ds)
    r, _ = rho("R_knn_avg", "gen_gap")
    ax.set_xlabel(r"multi-D rigidity $R_{\rm knn}$")
    ax.set_ylabel("generalization gap")
    ax.set_title(f"(c) $R_{{knn}}$ vs gap  (rho={r:.2f})")
    ax.legend(fontsize=8)

    # (d) predictive power bar chart: simple vs accuracy-partialled rho
    ax = axes[1, 0]
    targets = [("ece", "ECE"), ("fgsm01", "FGSM eps=0.1"),
               ("gen_gap", "Gen. gap"), ("acc", "Accuracy")]
    preds = [("R_avg", "R (1D)"), ("R_gram_avg", "R (Gram)"), ("R_knn_avg", "R (kNN)")]
    xpos = np.arange(len(targets))
    width = 0.26
    for j, (pk, pl) in enumerate(preds):
        vals = []
        for tk, _ in targets:
            r_, _ = rho(pk, tk)
            vals.append(abs(r_))
        ax.bar(xpos + (j - 1) * width, vals, width, label=pl)
    ax.set_xticks(xpos); ax.set_xticklabels([t for _, t in targets])
    ax.set_ylabel("|Spearman rho|")
    ax.set_title("(d) simple correlations with behaviour")
    ax.legend(fontsize=8)

    # (e) accuracy-partialled (honest) correlations
    ax = axes[1, 1]
    for j, (pk, pl) in enumerate(preds):
        vals = []
        for tk, _ in targets:
            r_, p_ = cr.partial_spearman(df[pk].values, df[tk].values,
                                         df["acc"].values)
            vals.append(abs(r_))
        ax.bar(xpos + (j - 1) * width, vals, width, label=pl)
    ax.set_xticks(xpos); ax.set_xticklabels([t for _, t in targets])
    ax.set_ylabel(r"|partial Spearman| (| acc)")
    ax.set_title("(e) correlations partialling out accuracy")
    ax.legend(fontsize=8)

    # (f) mechanism: multi-D entropy profile of a trained CNN vs Poisson ref
    ax = axes[1, 2]
    try:
        import analyze as az, train_networks as tn
        net, _ = tn.build_model(dict(dataset="mnist", arch="cnn2", dropout=0.0,
                                     init_scale=1.0, width=1, seed=0))
        net.load_state_dict(torch.load(os.path.join(RES, "nets", "model_mnist-cnn2-base.pt"),
                                       map_location="cpu"))
        net.eval()
        xs, _ = az.get_probe("mnist")
        reps = az.get_reps(net, xs)
        M, D = reps[-2].shape[0], 12
        kvals, Hx, means, radii = ep.knn_entropy_profile(
            ep.whiten_pca(reps[-2], D_eff=D))
        ref = ep.reference_knn_entropy(M, D, kvals, seed=0)
        ax.plot(kvals, Hx, "o-", color=C_TRAIN, label="trained CNN penult. layer")
        ax.plot(kvals, ref, "s--", color=C_POIS, label="i.i.d. Gaussian reference")
        ax.set_xlabel(r"nearest neighbours k")
        ax.set_ylabel(r"$H_x(k)$ (nats)")
        ax.set_title("(f) kNN entropy profile (whitened, matched radius)")
        ax.legend(fontsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f"unavailable:\n{e}", ha="center", fontsize=8)

    fig.tight_layout()
    save(fig, "fig9_multid_rigidity.png")
    plt.close(fig)


def fig10():
    """Causal dose-response: rigidity vs spread penalties close the gap?"""
    js = os.path.join(RES, "causal", "causal_mlp2_n1500_e25.json")
    with open(js) as f:
        d = json.load(f)
    lams = [0.0, 0.01, 0.1, 0.5, 1.0]
    arms = {"penultimate": "rigidity penalty",
            "spread": "spread (spectral-only) control"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for j, (arm, lab) in enumerate(arms.items()):
        gs, ts, sd = [], [], []
        for lam in lams:
            rs = [r for r in d["runs"] if r["arm"] == arm and r["lam"] == lam]
            gs.append(np.mean([r["gen_gap"] for r in rs]))
            ts.append(np.mean([r["test_acc_final"] for r in rs]))
            sd.append(np.std([r["gen_gap"] for r in rs]))
        c = C_TRAIN if arm == "penultimate" else C_LAT
        mk = "o" if arm == "penultimate" else "s"
        axes[0].errorbar(lams, gs, yerr=sd, marker=mk, capsize=4, lw=2,
                         c=c, label=lab)
        axes[1].plot(lams, ts, marker=mk, lw=2, c=c, label=lab)
    for ax, ylab, tit in [
            (axes[0], "generalization gap (train $-$ test)",
             "(a) Generalization gap vs. penalty dose"),
            (axes[1], "test accuracy",
             "(b) Test accuracy vs. penalty dose")]:
        ax.set_xscale("log", base=10)
        ax.set_xlabel(r"penalty strength $\lambda$")
        ax.set_ylabel(ylab)
        ax.set_title(tit)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig10_causal.png")
    plt.close(fig)


def main():
    which = sys.argv[1:] or ["all"]
    fns = dict(fig1=fig1, fig2=fig2, fig3=fig3, scatter=scatter,
               fig7=fig7, fig8=fig8, fig9=fig9, fig10=fig10)
    if "all" in which or "fig1" in which:
        print("fig1"); fig1()
    if "all" in which or "fig2" in which:
        print("fig2"); fig2()
    if "all" in which or "fig3" in which:
        print("fig3"); fig3()
    if "all" in which or "fig9" in which:
        print("fig9"); fig9()
    if "all" in which or "fig10" in which:
        print("fig10"); fig10()
    if "all" in which or "scatter" in which:
        print("scatters")
        scatter("gen_gap", "Generalization gap (train - test)",
                "fig4_gen_gap.png", use="R_avg")
        scatter("gen_gap", "Generalization gap (train - test)",
                "fig4b_gen_gap_gram.png", use="R_gram_avg")
        scatter("ece", "expected calibration error",
                "fig5_calibration.png", use="R_avg")
        scatter("fgsm03", "FGSM-robust accuracy (eps=0.3)",
                "fig6b_fgsm.png", use="R_avg")
        scatter("acc", "test accuracy", "fig6c_acc.png", use="R_avg")
    if "all" in which or "fig7" in which:
        print("fig7"); fig7()
    if "all" in which or "fig8" in which:
        print("fig8"); fig8()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()

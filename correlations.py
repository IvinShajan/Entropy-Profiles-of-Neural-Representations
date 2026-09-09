"""Correlation analysis: rigidity vs generalization / calibration / robustness.

Loads results/analysis/*.json, assembles a per-network table, and computes
Spearman / Pearson correlations between the entropy-profile rigidity measures
and behavioural metrics, alongside moment-based baselines (CKA, effective
rank) and a permutation null.

Usage: python3 correlations.py
"""
from __future__ import annotations
import os, json, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
AN = os.path.join(RES, "analysis")


def load_rows():
    rows = []
    for fn in sorted(os.listdir(AN)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(AN, fn)))
        row = {"label": d["label"]}
        row["arch"] = d["cfg"]["arch"]
        row["dataset"] = d["cfg"]["dataset"]
        row["acc"] = d["metrics"]["acc"]
        row["ece"] = d["metrics"]["ece"]
        row["fgsm01"] = d["metrics"]["fgsm01"]
        row["fgsm03"] = d["metrics"]["fgsm03"]
        row["pgd"] = d["metrics"]["pgd"]
        row["gen_gap"] = d.get("gen_gap", np.nan)
        layers = d["layers"]
        row["n_layers"] = len(layers)
        rp = [l["R_pc1"] for l in layers]
        rgram = [l["R_gram"] for l in layers]
        rvec = [l["R_vec"] for l in layers]
        rknn = [l.get("R_knn", np.nan) for l in layers]
        rknng = [l.get("R_knn_generic", np.nan) for l in layers]
        effr = [l["eff_rank"] for l in layers]
        row["R_avg"] = float(np.mean(rp))
        row["R_last"] = float(rp[-1])
        row["R_first"] = float(rp[0])
        row["R_max"] = float(np.max(rp))
        row["R_gram_avg"] = float(np.mean(rgram))
        row["R_vec_avg"] = float(np.nanmean(rvec))
        row["R_vec_last"] = float(np.nanmean([l["R_vec"] for l in layers[-1:]]))
        row["R_knn_avg"] = float(np.nanmean(rknn))
        row["R_knn_last"] = float(np.nanmean([l.get("R_knn", np.nan) for l in layers[-1:]]))
        row["R_knn_generic_avg"] = float(np.nanmean(rknng))
        row["R_knn_generic_last"] = float(np.nanmean([l.get("R_knn_generic", np.nan) for l in layers[-1:]]))
        row["effrank_avg"] = float(np.mean(effr))
        row["effrank_last"] = float(effr[-1])
        row["cka_avg"] = float(np.mean(d["cka_chain"])) if d["cka_chain"] else np.nan
        row["cka_io"] = float(d["R_input_layer"])
        row["layer_Rs"] = rp
        # gap stats & llr averaged over layers
        row["llr_avg"] = float(np.mean([l["llr_pc1"] for l in layers]))
        row["gapvar_avg"] = float(np.mean([l["gap_var_pc1"] for l in layers]))
        row["slope_avg"] = float(np.mean([l["slope_pc1"] for l in layers]))
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_p(x, y):
    from scipy.stats import spearmanr
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, np.nan
    rho, p = spearmanr(x[m], y[m])
    return rho, p


def spearman_boot_ci(x, y, n_boot=2000, seed=0, alpha=0.05):
    """Bootstrap 95% CI for the Spearman correlation (paired resampling)."""
    from scipy.stats import spearmanr
    m = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[m], y[m]
    if len(xv) < 8:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(xv)
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rhos[b] = spearmanr(xv[idx], yv[idx])[0]
    lo, hi = np.percentile(rhos, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def pearson_p(x, y):
    from scipy.stats import pearsonr
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, np.nan
    r, p = pearsonr(x[m], y[m])
    return r, p


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank-based)."""
    from scipy.stats import rankdata
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 8:
        return np.nan, np.nan
    rx, ry, rz = (rankdata(v[m]) for v in (x, y, z))
    X = np.column_stack([np.ones(m.sum()), rz])
    rxr = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ryr = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    r = np.corrcoef(rxr, ryr)[0, 1]
    n = m.sum()
    t = r * np.sqrt((n - 3) / max(1e-9, 1 - r ** 2))
    from scipy.stats import t as tdist
    p = 2 * tdist.sf(abs(t), n - 3)
    return float(r), float(p)


def partial_spearman_mult(x, y, Z):
    """Spearman partial correlation of x,y jointly controlling for the columns
    of Z (rank-based residualization)."""
    from scipy.stats import rankdata
    Z = np.asarray(Z, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    if m.sum() < 10:
        return np.nan, np.nan
    rx = rankdata(x[m])
    ry = rankdata(y[m])
    X = np.column_stack([np.ones(m.sum())] +
                        [rankdata(Z[m, j]) for j in range(Z.shape[1])])
    rxr = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ryr = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    r = np.corrcoef(rxr, ryr)[0, 1]
    n = m.sum()
    k = X.shape[1]
    t = r * np.sqrt((n - k - 1) / max(1e-9, 1 - r ** 2))
    from scipy.stats import t as tdist
    p = 2 * tdist.sf(abs(t), n - k - 1)
    return float(r), float(p)


TARGETS = {"gen_gap": "Generalization gap (train-test)", "ece": "ECE",
           "fgsm01": "FGSM acc eps=0.1", "fgsm03": "FGSM acc eps=0.3",
           "pgd": "PGD acc eps=0.3"}
PREDICTORS = {"R_avg": "Rigidity (mean layer)", "R_last": "Rigidity (final layer)",
              "R_gram_avg": "Rigidity (Gram spectrum)",
              "R_vec_avg": "Rigidity (single-input coords)",
              "R_knn_avg": "Rigidity (multidim kNN, mean layer)",
              "R_knn_last": "Rigidity (multidim kNN, final layer)",
              "R_knn_generic_avg": "Rigidity (multidim kNN generic)",
              "cka_avg": "Linear CKA (consecutive)",
              "cka_io": "Linear CKA (input-last)",
              "effrank_avg": "Effective rank (mean)",
              "llr_avg": "Spacing log-likelihood ratio",
              "gapvar_avg": "Spacing variance",
              "slope_avg": "Entropy-profile slope"}


def main():
    df = load_rows()
    df.to_csv(os.path.join(RES, "networks_table.csv"), index=False)
    print(f"loaded {len(df)} networks")

    results = {}
    for tname, tlabel in TARGETS.items():
        for pname, plabel in PREDICTORS.items():
            rho, p = spearman_p(df[pname].values, df[tname].values)
            results[f"{pname}->{tname}"] = dict(rho=rho, p=p)
            if tname == "gen_gap" and pname in ("R_avg", "R_gram_avg", "R_knn_avg",
                                                "R_knn_generic_avg", "cka_io",
                                                "effrank_avg", "acc"):
                lo, hi = spearman_boot_ci(df[pname].values, df[tname].values)
                results[f"{pname}->{tname}"]["ci95"] = [lo, hi]
    # partial correlations controlling for test accuracy (confound check)
    RIG = ["R_avg", "R_last", "R_gram_avg", "R_knn_avg", "R_knn_generic_avg",
           "cka_avg", "effrank_avg", "cka_io"]
    for tname in TARGETS:
        for pname in RIG:
            r, p = partial_spearman(df[pname].values, df[tname].values,
                                    df["acc"].values)
            results[f"partial|acc:{pname}->{tname}"] = dict(rho=r, p=p)
            r, p = partial_spearman(df[pname].values, df[tname].values,
                                    df["dataset"].map({"mnist": 0, "fashion": 1,
                                                       "cifar-small": 2,
                                                       "cifar100": 3,
                                                       "svhn": 4}).values)
            results[f"partial|ds:{pname}->{tname}"] = dict(rho=r, p=p)
        # does rigidity add information beyond effective rank / CKA baselines?
        for pname in ["R_avg", "R_gram_avg", "R_knn_avg"]:
            r, p = partial_spearman(df[pname].values, df[tname].values,
                                    df["effrank_avg"].values)
            results[f"partial|effrank:{pname}->{tname}"] = dict(rho=r, p=p)
            r, p = partial_spearman(df[pname].values, df[tname].values,
                                    df["cka_io"].values)
            results[f"partial|cka_io:{pname}->{tname}"] = dict(rho=r, p=p)
            Z = np.column_stack([df["acc"].values, df["effrank_avg"].values])
            r, p = partial_spearman_mult(df[pname].values, df[tname].values, Z)
            results[f"partial|acc+eff:{pname}->{tname}"] = dict(rho=r, p=p)
    # also within-dataset (MNIST population only, n>20)
    mn = df[df.dataset == "mnist"]
    results["_n_mnist"] = len(mn)
    for tname in TARGETS:
        for pname in RIG:
            rho, p = spearman_p(mn[pname].values, mn[tname].values)
            results[f"mnist:{pname}->{tname}"] = dict(rho=rho, p=p)
        for pname in ["R_avg", "R_gram_avg", "R_knn_avg"]:
            r, p = partial_spearman(mn[pname].values, mn[tname].values,
                                    mn["acc"].values)
            results[f"mnist+partial|acc:{pname}->{tname}"] = dict(rho=r, p=p)

    # permutation null for the headline correlations (1D and multidim)
    rng = np.random.default_rng(0)
    for pk in ["R_avg", "R_gram_avg", "R_knn_avg", "R_knn_generic_avg"]:
        x = df[pk].values
        y = df["gen_gap"].values
        m = np.isfinite(x) & np.isfinite(y)
        obs = spearman_p(x, y)[0]
        null = []
        for _ in range(2000):
            null.append(spearman_p(x[m], rng.permutation(y[m]))[0])
        results[f"_perm_null_{pk}_gen_gap"] = dict(
            obs=obs, p_perm=float(np.mean(np.abs(null) >= abs(obs))))

    with open(os.path.join(RES, "correlations.json"), "w") as f:
        json.dump(results, f, indent=1)

    # pretty table
    print("\n" + "=" * 100)
    print(f"{'predictor':<34} {'target':<30} {'rho':>7} {'p':>9}")
    print("=" * 100)
    for k, v in results.items():
        if k.startswith("_") or k.startswith("mnist:"):
            continue
        if not k.startswith("R_") and not k.startswith("llr") and \
           not k.startswith("gapvar") and not k.startswith("slope"):
            continue
        rho, p = v["rho"], v["p"]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        pred, tgt = k.split("->")
        print(f"{pred:<34} {tgt:<30} {rho:7.3f} {p:9.2e} {stars}")

    print("\nMNIST-only correlations (n=%d):" % results["_n_mnist"])
    for k, v in results.items():
        if k.startswith("mnist:"):
            rho, p = v["rho"], v["p"]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {k[6:]:<20} rho={rho:7.3f} p={p:9.2e} {stars}")


if __name__ == "__main__":
    main()

"""Entropy profiles of 1D point configurations.

Core math for the study. A "point configuration" is a finite set of real
numbers (e.g. the PC-1 projections of a batch of layer activations, or the
unfolded spectrum of a Gram matrix). We measure how the *entropy of the
sliding-window counting statistics* depends on window length L, and compare
the resulting profile against three reference ensembles:

  * Poisson (generic / random): gaps ~ Exp(1).
  * GUE    (intermediate, RMT): eigenvalues of a random Hermitian matrix.
  * Lattice (rigid): perfectly ordered points.

The deviation of an empirical profile from the Poisson and Lattice references
defines the *Rigidity Index* R in [0,1] (0 = generic, ~0.6 = GUE-like,
1 = lattice-like).

All entropies are Shannon entropies in natural logarithms (nats).
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

EPS = 1e-12


# ----------------------------------------------------------------------------
# Reference ensembles (unit mean spacing)
# ----------------------------------------------------------------------------

def poisson_points(M: int, seed: int | None = None) -> np.ndarray:
    """Poisson process on the line: gaps ~ Exp(1). Unit mean spacing."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0, M)
    return np.cumsum(gaps)


def lattice_points(M: int, jitter: float = 0.0, seed: int | None = None) -> np.ndarray:
    """Perfect lattice (rigid) or jittered lattice. Unit mean spacing."""
    x = np.arange(M, dtype=float)
    if jitter > 0:
        rng = np.random.default_rng(seed)
        x = x + rng.uniform(-jitter, jitter, M)
    return np.sort(x)


def gue_points(M: int, seed: int | None = None) -> np.ndarray:
    """GUE eigenvalue spacings (unfolded, unit mean spacing).

    Samples a random Hermitian matrix, takes eigenvalues, unfolds with the
    semicircle CDF (radius fixed empirically from the sample), returns the
    unfolded eigenvalues whose local spacings follow Wigner-Dyson statistics.
    """
    rng = np.random.default_rng(seed)
    n = int(np.ceil(2 * np.sqrt(M)))
    n = max(n, 256)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    H = (A + 1j * B)
    H = (H + H.conj().T) / 2.0
    lam = np.linalg.eigvalsh(H).real
    R = float(2.0 * np.sqrt(np.mean(lam**2))) + EPS
    # semicircle CDF on [-R, R]
    t = np.clip(lam / R, -1, 1)
    F = 0.5 + (1.0 / np.pi) * (t * np.sqrt(np.maximum(0.0, 1 - t * t)) + np.arcsin(t))
    u = F * n
    x = np.sort(u)
    # drop the extreme edges where the semicircle density is unstable
    lo, hi = np.searchsorted(x, 0.05 * n), np.searchsorted(x, 0.95 * n)
    return np.sort(x[lo:hi])


# ----------------------------------------------------------------------------
# Core statistics
# ----------------------------------------------------------------------------

def normalize_gaps(x: np.ndarray) -> np.ndarray:
    """Sorted nearest-neighbour gaps, rescaled to unit mean."""
    x = np.sort(np.asarray(x, dtype=float))
    g = np.diff(x)
    m = g.mean()
    return g / m if m > EPS else g


def window_counts(x: np.ndarray, L: float, method: str = "midpoints") -> np.ndarray:
    """Counts of points in sliding windows of length L (in mean-spacing units).

    Windows are centred at the midpoints between consecutive sorted points,
    so every window lies interior to the configuration.
    """
    x = np.sort(np.asarray(x, dtype=float))
    centers = (x[:-1] + x[1:]) / 2.0
    centers = centers[centers + L <= x[-1]]
    if centers.size == 0:
        return np.zeros(0)
    right = np.searchsorted(x, centers + L, side="right")
    left = np.searchsorted(x, centers, side="left")
    return (right - left).astype(float)


def shannon_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        return 0.0
    p = np.bincount(counts.astype(int)).astype(float)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def window_entropy(x: np.ndarray, L: float) -> float:
    """H(L): Shannon entropy (nats) of the window-count distribution."""
    return shannon_entropy(window_counts(x, L))


def number_variance(x: np.ndarray, L: float) -> float:
    """Sigma^2(L): variance of the window-count distribution (unfolded)."""
    c = window_counts(x, L)
    return float(c.var())


def entropy_profile(x: np.ndarray, Ls: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Entropy profile H(L) over a log-spaced grid of window lengths."""
    if Ls is None:
        Ls = np.logspace(np.log10(2.0), np.log10(64.0), 10)
    Hs = np.array([window_entropy(x, L) for L in Ls])
    return np.asarray(Ls, dtype=float), Hs


def number_variance_profile(x: np.ndarray, Ls: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if Ls is None:
        Ls = np.logspace(np.log10(2.0), np.log10(64.0), 10)
    Vs = np.array([number_variance(x, L) for L in Ls])
    return np.asarray(Ls, dtype=float), Vs


def delta3(x: np.ndarray, L: float, step: float = 0.25) -> float:
    """Dyson-Mehta rigidity Delta_3(L).

    For each window of length L, least-squares deviation of the counting
    function N(s) from its best linear fit, averaged over windows.
    """
    x = np.sort(np.asarray(x, dtype=float))
    L = float(L)
    mids = (x[:-1] + x[1:]) / 2.0
    vals = []
    for t in mids:
        seg = x[(x >= t) & (x <= t + L)]
        if seg.size < 3:
            continue
        s = np.arange(t, t + L, step)
        Ns = np.searchsorted(seg, s, side="right")
        # least-squares line fit
        s2, N2 = s.mean(), Ns.mean()
        a = np.sum((s - s2) * (Ns - N2)) / (np.sum((s - s2) ** 2) + EPS)
        b = N2 - a * s2
        res = np.mean((Ns - (a * s + b)) ** 2)
        vals.append(res)
    if not vals:
        return np.nan
    return float(np.mean(vals))


# ----------------------------------------------------------------------------
# Reference comparison
# ----------------------------------------------------------------------------

WIGNER_CDF = lambda s: 1.0 - np.exp(-np.pi * s**2 / 4.0)
WIGNER_PDF = lambda s: (np.pi * s / 2.0) * np.exp(-np.pi * s**2 / 4.0)


def reference_stats(seed: int = 0):
    """Precompute empirical Poisson/GUE/lattice curves for a fixed M."""
    pass


def ks_poisson(g: np.ndarray) -> tuple[float, float]:
    """KS test of normalized gaps vs Exp(1) (Poisson reference)."""
    g = np.asarray(g, dtype=float)
    return stats.kstest(g, "expon", args=(0.0, 1.0))


def ks_gue(g: np.ndarray) -> tuple[float, float]:
    """KS test vs the Wigner surmise (GUE reference, GOE-GUE identical surmise)."""
    g = np.asarray(g, dtype=float)
    return stats.kstest(g, WIGNER_CDF)


def log_likelihood_ratio(g: np.ndarray) -> float:
    """mean log p_Wigner / p_Poisson per gap (positive = GUE-like)."""
    g = np.asarray(g, dtype=float)
    g = g[g > 0]
    return float(np.mean(np.log(WIGNER_PDF(g) / np.exp(-g))))


def gap_var(g: np.ndarray) -> float:
    g = np.asarray(g, dtype=float)
    return float(g.var())


# ----------------------------------------------------------------------------
# Unfolding (density removal)
# ----------------------------------------------------------------------------

def unfold_kde(x: np.ndarray, bw_mult: float = 20.0) -> np.ndarray:
    """Unfold a 1D point set by smoothing the empirical CDF with a Gaussian
    kernel whose bandwidth is proportional to the *local mean spacing*.

    The smoothing scale is set large relative to the spacing (so Poisson /
    rigidity fluctuations survive) but small relative to the density's
    variation scale (so genuine density gradients are removed). Returns the
    unfolded coordinates with unit local mean spacing.
    """
    from scipy.special import erf
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n < 8:
        return np.sort(np.linspace(0, n, n))
    spacing = float(np.mean(np.diff(x)))
    h = bw_mult * spacing
    # F(x) = (1/n) sum_j Phi((x - x_j)/h)
    X = np.broadcast_to(x, (n, n))
    D = (X - X.T) / h
    F = 0.5 * (1.0 + erf(D / np.sqrt(2.0)))
    F = F.mean(axis=1)
    u = F * n
    u = np.sort(u)
    # force unit mean spacing after unfolding
    u = u - u.mean()
    u = u * (n - 1) / (u[-1] - u[0])
    return u


# ----------------------------------------------------------------------------
# Rigidity index
# ----------------------------------------------------------------------------

def rigidity_index(x: np.ndarray,
                   Lmin: float = 4.0, Lmax: float = 32.0,
                   nLs: int = 6, nref: int = 6, seed: int = 0,
                   return_parts: bool = False):
    """Rigidity Index R in [0,1].

    R = 1 - < (H_x(L) - H_lat(L)) / (H_pois(L) - H_lat(L)) >_L
    with H_pois, H_lat the empirical window-entropy curves of Poisson and
    lattice configurations with the same number of points.

    R ~ 0 : generic / random (Poisson-like)
    R ~ 0.6 : intermediate (GUE-like)
    R ~ 1 : rigid (lattice-like)
    """
    x = np.sort(np.asarray(x, dtype=float))
    if x.size < 32:
        raise ValueError("too few points for a meaningful profile")
    M = x.size
    Ls = np.linspace(Lmin, Lmax, nLs)
    Hx = np.array([window_entropy(x, L) for L in Ls])

    Hp = np.zeros(nLs)
    for i in range(nref):
        xp = poisson_points(M, seed=seed + i)
        for j, L in enumerate(Ls):
            Hp[j] += window_entropy(xp, L) / nref

    xl = lattice_points(M)
    Hl = np.array([window_entropy(xl, L) for L in Ls])

    denom = np.maximum(Hp - Hl, EPS)
    frac = (Hx - Hl) / denom
    R = 1.0 - np.mean(frac)
    R = float(np.clip(R, 0.0, 1.0))
    if return_parts:
        return R, Hx, Hp, Hl, Ls
    return R


def profile_slope(x: np.ndarray, Lmin: float = 8.0, Lmax: float = 64.0, nLs: int = 8) -> float:
    """dH/d ln L over the asymptotic window; ~1/2 for Poisson, ~0 for lattice."""
    Ls = np.logspace(np.log10(Lmin), np.log10(Lmax), nLs)
    _, Hs = entropy_profile(x, Ls)
    A = np.vstack([np.ones(nLs), np.log(Ls)]).T
    coef, *_ = np.linalg.lstsq(A, Hs, rcond=None)
    return float(coef[1])


# ----------------------------------------------------------------------------
# Multi-dimensional entropy profiles (k-NN ball counting)
# ----------------------------------------------------------------------------

def knn_ball_counts(X: np.ndarray, r: float, metric: str = "euclidean") -> np.ndarray:
    """Count points within a *fixed* radius r of each point (D-dim window count).

    This is the multi-dimensional analog of the 1D sliding-window count: for a
    Poisson point process, counts ~ Poisson(lambda V(r)) so mean == variance;
    for a rigid (lattice-like) arrangement, counts are nearly identical across
    points; for a clustered arrangement they are over-dispersed.

    Args:
        X: (N, D) array of N points in D dimensions
        r: fixed radius (in the metric's units)
        metric: distance metric ('euclidean' or 'cosine')

    Returns:
        Array of N counts (each point counts itself + neighbours within r)
    """
    from scipy.spatial.distance import cdist

    N = X.shape[0]
    if N < 2 or r <= 0:
        return np.ones(N, dtype=float)

    if metric == "cosine":
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        D = cdist(X_norm, X_norm, metric="cosine")
    else:
        D = cdist(X, X, metric="euclidean")
    np.fill_diagonal(D, 0.0)

    counts = np.sum(D <= r, axis=1)
    return counts.astype(float)


def knn_entropy_profile(X: np.ndarray, k_vals: list[int] | None = None,
                        metric: str = "euclidean",
                        r_quantile: float = 0.5) -> tuple[list[int], np.ndarray, np.ndarray, list[float]]:
    """Entropy profile over k-NN neighborhood scales.

    For each k, the *median k-th-nearest-neighbour distance* of the point set
    fixes a radius r_k such that a typical window contains ~k+1 points (the
    D-dimensional analog of a 1D window of length L in mean-spacing units).
    The ball-count distribution at radius r_k yields:
      H_x(k) : entropy of the counts
      m(k)   : mean count
    which the rigidity index compares against a matched Poisson reference.

    Args:
        X: (N, D) array of points
        k_vals: list of k values to test (default: log-spaced)
        metric: distance metric
        r_quantile: quantile of the k-NN distance distribution used as radius

    Returns:
        k_vals, H(k) array, mean-count array, radii r_k
    """
    from scipy.spatial.distance import cdist

    N, D = X.shape
    if k_vals is None:
        k_vals = [2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 200]
        k_vals = [k for k in k_vals if 0 < k < N - 1]
    if not k_vals:
        return [], np.array([]), np.array([]), []

    if metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        DM = cdist(Xn, Xn, metric="cosine")
    else:
        DM = cdist(X, X, metric="euclidean")
    np.fill_diagonal(DM, np.inf)

    Hs, means, radii = [], [], []
    for k in k_vals:
        knn_d = np.partition(DM, k, axis=1)[:, k]   # k-th NN distance (0-indexed)
        r_k = float(np.quantile(knn_d, r_quantile))
        counts = np.sum(DM <= r_k, axis=1)
        Hs.append(shannon_entropy(counts))
        means.append(float(counts.mean()))
        radii.append(r_k)
    return k_vals, np.array(Hs), np.array(means), radii


def reference_knn_entropy(M: int, D: int, k_vals: list[int],
                          nref: int = 6, seed: int = 0,
                          metric: str = "euclidean",
                          r_quantile: float = 0.5) -> np.ndarray:
    """Poisson reference ball-count entropies, matched radius rule.

    For each k, draws nref i.i.d. standard-normal clouds of (M, D) points and
    computes the ball-count entropy at that sample's OWN median k-th-NN
    distance (the same radius rule as the empirical profile). Because both the
    empirical and the reference profiles use the identical rule, an i.i.d.
    configuration scores R ~ 0 by construction. A lattice (rigid) reference
    has ball-count entropy 0 at every scale and is implicit.

    Returns:
        H_poisson array of shape (len(k_vals),)
    """
    rng = np.random.default_rng(seed)
    Hp = np.zeros(len(k_vals))
    for i in range(nref):
        Xp = rng.standard_normal((M, D))
        _, Hp_i, _, _ = knn_entropy_profile(Xp, k_vals, metric, r_quantile)
        Hp += Hp_i / nref
    return Hp


def rigidity_index_knn(X: np.ndarray,
                       k_vals: list[int] | None = None,
                       nref: int = 6,
                       seed: int = 0,
                       metric: str = "euclidean",
                       r_quantile: float = 0.5,
                       return_parts: bool = False):
    """Multi-dimensional Rigidity Index from k-NN ball-count entropy.

    For each k, the median k-th-NN distance fixes a window radius r_k so a
    typical window contains ~k+1 points. The empirical ball-count entropy
    H_x(k) is compared against the Poisson reference H_pois(k) computed with
    the *identical* radius rule on i.i.d. Gaussian clouds of matched (N, D):

        R = 1 - mean_k [ H_x(k) / H_pois(k) ]

    R ~ 0 : generic (i.i.d.-like, ball counts Poisson)
    R > 0 : sub-Poisson entropy (locally ordered / rigid)
    R < 0 : super-Poisson entropy (locally clustered / over-dispersed)

    Clamped to [0, 1] by default (R<0 reported as 0); use
    knn_poisson_generic_index for the sign-preserving variant.

    Args:
        X: (N, D) point configuration
        k_vals: neighborhood sizes to evaluate
        nref: number of Poisson reference samples
        seed: random seed
        metric: distance metric
        r_quantile: quantile of the k-NN distance distribution used as radius
        return_parts: if True, return (R, Hx, Hpois, means, k_vals)

    Returns:
        R in [0, 1], optionally with parts
    """
    M, D = X.shape
    if k_vals is None:
        k_vals = [2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 200]
        k_vals = [k for k in k_vals if 0 < k < M - 1]
    if not k_vals:
        return float("nan")

    _, Hx, means, _ = knn_entropy_profile(X, k_vals, metric, r_quantile)
    Hp = reference_knn_entropy(M, D, k_vals, nref, seed, metric, r_quantile)

    frac = np.mean(Hx / np.maximum(Hp, EPS))
    R = float(np.clip(1.0 - frac, 0.0, 1.0))

    if return_parts:
        return R, Hx, Hp, means, k_vals
    return R


def pairwise_distance_entropy(X: np.ndarray,
                              bins: int = 50,
                              metric: str = "euclidean",
                              sample_pairs: int | None = None) -> float:
    """Entropy of the pairwise distance distribution.

    Complementary to k-NN entropy: captures the global arrangement. For N
    points there are N(N-1)/2 pairwise distances.

    Args:
        X: (N, D) points
        bins: number of histogram bins
        metric: distance metric
        sample_pairs: if set, subsample this many pairs (for large N)

    Returns:
        Shannon entropy (nats) of the pairwise distance histogram
    """
    from scipy.spatial.distance import pdist

    N = X.shape[0]
    if N < 2:
        return 0.0

    dists = pdist(X, metric=metric)
    if sample_pairs and len(dists) > sample_pairs:
        rng = np.random.default_rng(0)
        dists = dists[rng.choice(len(dists), sample_pairs, replace=False)]

    dists = dists[dists > 1e-10]
    if len(dists) == 0:
        return 0.0

    hist, _ = np.histogram(dists, bins=bins, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    return float(-(hist * np.log(hist)).sum())


def reference_pairwise_entropy(M: int, D: int,
                               nref: int = 6,
                               seed: int = 0,
                               metric: str = "euclidean") -> tuple[float, float]:
    """Poisson and lattice reference pairwise distance entropies."""
    rng = np.random.default_rng(seed)
    H_poisson = 0.0
    for i in range(nref):
        X = rng.standard_normal((M, D))
        H_poisson += pairwise_distance_entropy(X, metric=metric) / nref
    side = int(np.ceil(M ** (1.0 / D)))
    grid = np.meshgrid(*[np.linspace(0, side - 1, side)] * D, indexing='ij')
    X_lat = np.stack([g.ravel() for g in grid], axis=1)[:M]
    H_lattice = pairwise_distance_entropy(X_lat, metric=metric)
    return H_poisson, H_lattice


def rigidity_index_pairwise(X: np.ndarray,
                            nref: int = 6,
                            seed: int = 0,
                            metric: str = "euclidean",
                            return_parts: bool = False) -> float:
    """Rigidity index from pairwise distance entropy."""
    M, D = X.shape
    Hx = pairwise_distance_entropy(X, metric=metric)
    Hp, Hl = reference_pairwise_entropy(M, D, nref, seed, metric)
    R = 1.0 - (Hx - Hl) / max(Hp - Hl, EPS)
    R = float(np.clip(R, 0.0, 1.0))
    if return_parts:
        return R, Hx, Hp, Hl
    return R


def whiten_pca(X: np.ndarray, D_eff: int = 12) -> np.ndarray:
    """Center + PCA-whiten to D_eff dimensions (unit variance, orthogonal).

    High-dimensional activations are projected to their top-D_eff principal
    components and whitened so that Euclidean distances are comparable with
    an isotropic Poisson reference before the multi-dimensional rigidity index
    is computed. Rank-deficient inputs are padded with a tiny isotropic
    regularisation so the effective dimension never collapses."""
    Xc = X - X.mean(0, keepdims=True)
    k = min(D_eff, Xc.shape[1], Xc.shape[0] - 1)
    if k <= 0:
        return np.zeros((Xc.shape[0], 1))
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    W = U[:, :k] * S[:k]
    W = W / (W.std(0, keepdims=True) + EPS)
    if W.shape[1] < D_eff and W.shape[0] > 1:
        pad = np.random.default_rng(0).standard_normal(
            (W.shape[0], D_eff - W.shape[1])) * 1e-8
        W = np.hstack([W, pad])
    return W


def rigidity_index_knn_whitened(X: np.ndarray,
                                D_eff: int = 12,
                                k_vals: list[int] | None = None,
                                nref: int = 6,
                                seed: int = 0,
                                metric: str = "euclidean",
                                return_parts: bool = False):
    """Multi-dimensional (whitened) rigidity index.

    Whitens activations to D_eff PCA dimensions, then computes the k-NN
    ball-count entropy profile and the Poisson-calibrated index (matched
    radius rule). R ~ 0 : generic / random;  R > 0 : rigid (lattice-like).
    """
    W = whiten_pca(X, D_eff)
    return rigidity_index_knn(W, k_vals=k_vals, nref=nref, seed=seed,
                              metric=metric, return_parts=return_parts)


def knn_poisson_generic_index(X: np.ndarray,
                              D_eff: int = 12,
                              k_vals: list[int] | None = None,
                              nref: int = 6,
                              seed: int = 0,
                              metric: str = "euclidean") -> float:
    """Multi-dimensional 'distance from generic' index (sign-preserving).

    R_g = 1 - mean_k [H_x(k) / H_pois(k)], 0 = exactly generic; positive =
    sub-Poisson entropy (ordered / rigid); negative = super-Poisson entropy
    (clustered / over-dispersed). Unlike the clamped [0,1] index, this keeps
    the sign so clustering is not conflated with rigidity."""
    W = whiten_pca(X, D_eff)
    M, D = W.shape
    if k_vals is None:
        k_vals = [2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 200]
        k_vals = [k for k in k_vals if 0 < k < M - 1]
    if not k_vals:
        return float("nan")
    _, Hx, _, _ = knn_entropy_profile(W, k_vals, metric)
    Hp = reference_knn_entropy(M, D, k_vals, nref, seed, metric)
    return float(1.0 - np.mean(Hx / np.maximum(Hp, EPS)))


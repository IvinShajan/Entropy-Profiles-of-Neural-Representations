"""Sanity checks for entropy_profiles.py.

Verifies that synthetic configurations reproduce the known fingerprint:
  * Poisson  -> gaps ~ Exp(1); H(L) ~ 1/2 ln(2 pi e L); R ~ 0
  * Lattice  -> H(L) ~ 0; R ~ 1; slope ~ 0
  * GUE      -> number variance ~ (2/pi^2) ln L; R in (0.4, 0.8); slope in between
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import entropy_profiles as ep

def main():
    M = 1200
    print("=" * 78)
    for name, x in [("POISSON", ep.poisson_points(M, seed=1)),
                    ("GUE", ep.gue_points(M, seed=1)),
                    ("LATTICE", ep.lattice_points(M))]:
        x = np.sort(x)
        g = ep.normalize_gaps(x)
        ks_p = ep.ks_poisson(g)
        ks_g = ep.ks_gue(g)
        Ls = np.logspace(np.log10(2), np.log10(64), 10)
        _, Hs = ep.entropy_profile(x, Ls)
        _, Vs = ep.number_variance_profile(x, Ls)
        R = ep.rigidity_index(x, seed=0)
        slope = ep.profile_slope(x)
        # theory checks at L=20
        i20 = np.argmin(np.abs(Ls - 20))
        H20, V20 = Hs[i20], Vs[i20]
        theo_H_pois = 0.5 * np.log(2 * np.pi * np.e * 20)
        print(f"\n[{name}]")
        print(f"  ks vs Exp(1)     : stat={ks_p[0]:.4f} p={ks_p[1]:.3f} (expect Poisson: p>0.05)")
        print(f"  ks vs Wigner     : stat={ks_g[0]:.4f} p={ks_g[1]:.3f} (expect GUE: p>0.05)")
        print(f"  H(20)={H20:6.3f}   Poisson theory 1/2 ln(2pieL)={theo_H_pois:6.3f}")
        print(f"  V(20)={V20:6.3f}   Poisson V~20; GUE V~{(2/np.pi**2)*np.log(20):.3f}")
        print(f"  slope dH/dlnL={slope:5.3f}  (Poisson~0.5, lattice~0)")
        print(f"  rigidity R={R:.3f}  (Poisson~0, GUE~0.6, lattice~1)")
        # Delta3: Poisson ~ L/15, lattice ~ small, GUE ~ (1/pi^2) ln L
        d3 = ep.delta3(x, 20.0)
        print(f"  Delta3(20)={d3:6.4f}  (Poisson~1.33, GUE~0.30, lattice~0)")

    # log-likelihood ratio sanity
    gp = ep.normalize_gaps(ep.poisson_points(4000, seed=2))
    gg = ep.normalize_gaps(ep.gue_points(4000, seed=2))
    gl = ep.normalize_gaps(ep.lattice_points(4000))
    print("\nLLR (pos=GUE-like): Poisson=%.3f GUE=%.3f Lattice=%.3f"
          % (ep.log_likelihood_ratio(gp), ep.log_likelihood_ratio(gg), ep.log_likelihood_ratio(gl)))
    print("gap var: Poisson=%.3f GUE=%.3f Lattice=%.3f"
          % (ep.gap_var(gp), ep.gap_var(gg), ep.gap_var(gl)))

    # KDE unfolding: must preserve Poisson, GUE, and lattice signatures
    for name, x in [("POISSON", ep.poisson_points(1200, seed=3)),
                    ("GUE", ep.gue_points(1200, seed=3)),
                    ("LATTICE", ep.lattice_points(1200))]:
        xu = ep.unfold_kde(x)
        gu = ep.normalize_gaps(xu)
        print(f"\nunfold({name}) ks vs Exp(1) : {ep.ks_poisson(gu)[0]:.3f} p={ep.ks_poisson(gu)[1]:.3f}")
        print(f"unfold({name}) ks vs Wigner : {ep.ks_gue(gu)[0]:.3f} p={ep.ks_gue(gu)[1]:.3f}")
        print(f"unfold({name}) R            : {ep.rigidity_index(xu):.3f}")
    # i.i.d. Gaussian samples (the "generic high-dim" reference) should unfold to Poisson
    rng = np.random.default_rng(7)
    xg = np.sort(rng.standard_normal(1200))
    xgu = ep.unfold_kde(xg)
    print(f"\nunfold(gauss)  ks vs Exp(1) : {ep.ks_poisson(ep.normalize_gaps(xgu))}")
    print("ALL SYNTHETIC CHECKS DONE")

if __name__ == "__main__":
    main()

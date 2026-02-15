import numpy as np
from diag import evals_rel, f_res_bare

def _peak_spacing_from_pairwise_diffs(energies, min_diff=1.0, bin_width=0.05):
    energies = np.array(sorted(energies))
    diffs = []
    for i in range(len(energies)):
        for j in range(i + 1, len(energies)):
            d = energies[j] - energies[i]
            if d >= min_diff:
                diffs.append(df := float(d))
    diffs = np.array(diffs)
    if diffs.size == 0:
        raise ValueError("Not enough diffs to estimate a spacing.")

    dmin, dmax = diffs.min(), diffs.max()
    nbins = max(10, int(np.ceil((dmax - dmin) / bin_width)))
    hist, edges = np.histogram(diffs, bins=nbins, range=(dmin, dmax))
    k = int(np.argmax(hist))
    return 0.5 * (edges[k] + edges[k+1])


def _nearest_int_multiple_residual(E, omega):
    m = int(np.round(E / omega))
    return abs(E - m * omega), m


def fit_effective_params_2mode(evals_rel, tol_nonmultiple=0.20, omega_r_known=None):
    E = np.array(sorted(evals_rel))
    if E[0] != 0.0:
        E = E - E[0]

    if omega_r_known is not None:
        omega_r = float(omega_r_known)
    else:
        omega_r = _peak_spacing_from_pairwise_diffs(E[:min(len(E), 16)], min_diff=1.0, bin_width=0.02)

    omega_q = None
    for Ek in E[1:]:
        resid, m = _nearest_int_multiple_residual(Ek, omega_r)
        if resid > tol_nonmultiple:
            omega_q = float(Ek)
            break
    if omega_q is None:
        omega_q = float(E[2]) if len(E) > 2 else float(E[1])

    target_2q = 2.0 * omega_q
    idx_2q = int(np.argmin(np.abs(E - target_2q)))
    alpha_q = float(E[idx_2q] - target_2q)

    target_11 = omega_q + omega_r
    idx_11 = int(np.argmin(np.abs(E - target_11)))
    chi_qr = float(E[idx_11] - target_11)

    return dict(omega_r=omega_r, omega_q=omega_q, alpha_q=alpha_q, chi_qr=chi_qr)


def predicted_energy_2mode(nq, nr, p, include_chi=True):
    omega_q, omega_r = p["omega_q"], p["omega_r"]
    alpha_q = p["alpha_q"]
    chi_qr = p["chi_qr"] if include_chi else 0.0
    return (omega_q * nq + 0.5 * alpha_q * nq * (nq - 1) + omega_r * nr + chi_qr * nq * nr)


def assign_labels_2mode(evals_rel, p, nq_max=6, nr_max=10, include_chi=True):
    E = np.array(sorted(evals_rel))
    E = E - E[0]

    cand = []
    for nq in range(nq_max + 1):
        for nr in range(nr_max + 1):
            Ep = predicted_energy_2mode(nq, nr, p, include_chi=include_chi)
            cand.append((Ep, nq, nr))
    cand.sort(key=lambda t: t[0])

    used = set()
    out = []
    for k, Ek in enumerate(E):
        best = None
        best_score = 1e99

        for Ep, nq, nr in cand:
            if (nq, nr) in used:
                continue
            if abs(Ep - Ek) > 1.0:
                continue

            resid = abs(Ep - Ek)
            penalty = 1e-3 * (nq + nr)
            score = resid + penalty
            if score < best_score:
                best_score = score
                best = (nq, nr, Ep, resid)

        if best is None:
            for Ep, nq, nr in cand:
                if (nq, nr) in used:
                    continue
                resid = abs(Ep - Ek)
                penalty = 1e-3 * (nq + nr)
                score = resid + penalty
                if score < best_score:
                    best_score = score
                    best = (nq, nr, Ep, resid)

        nq, nr, Ep, resid = best
        used.add((nq, nr))
        out.append(dict(k=k, E=Ek, nq=nq, nr=nr, E_pred=Ep, resid=resid))

    return out


OMEGA_R_KNOWN = 6.2
_omega_r = OMEGA_R_KNOWN if OMEGA_R_KNOWN is not None else f_res_bare

params = fit_effective_params_2mode(evals_rel, omega_r_known=_omega_r)
assign_nochi = assign_labels_2mode(evals_rel, params, include_chi=False)
assign_chi   = assign_labels_2mode(evals_rel, params, include_chi=True)

print("\n[Fitted effective 2-mode parameters] (GHz)")
_r_src = " (fixed)" if OMEGA_R_KNOWN is not None else " (from circuit bare f_r)"
print(f"omega_r = {params['omega_r']:.6f}" + _r_src)
print(f"omega_q ≈ {params['omega_q']:.6f}")
print(f"alpha_q ≈ {params['alpha_q']:.6f}")
print(f"chi_qr ≈ {params['chi_qr']:.6f} (effective dispersive/hybridization shift)")

def _pretty_print(assignments, title):
    print(f"\n{title}")
    print("  idx    E_obs(GHz)      label      E_pred(GHz)     |Δ|(MHz)")
    for a in assignments:
        print(f"  |{a['k']:2d}>  {a['E']:11.6f}   |{a['nq']},{a['nr']}>   {a['E_pred']:11.6f}    {1e3*a['resid']:8.2f}")

_pretty_print(assign_nochi,"[Labeling using Kerr + harmonic only (no chi_qr)]")
_pretty_print(assign_chi, "[Labeling using Kerr + harmonic + effective chi_qr]")

def energy_of_label(assignments, nq, nr):
    for a in assignments:
        if a["nq"] == nq and a["nr"] == nr:
            return a["E"]
    return None

E00 = energy_of_label(assign_chi, 0, 0)
Eq10 = energy_of_label(assign_chi, 1, 0)
Er01 = energy_of_label(assign_chi, 0, 1)

print("\n[Fundamentals inferred from labels] (GHz)")
if Eq10 is not None:
    print(f"  f_q01 ≈ E|1,0> - E|0,0> = {Eq10 - E00:.6f}")
if Er01 is not None:
    print(f"  f_r01 ≈ E|0,1> - E|0,0> = {Er01 - E00:.6f}")
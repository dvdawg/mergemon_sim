import numpy as np
from diag import evals_rel, f_res_bare
from circuit_from_design import (
    get_qubit_params,
    get_ancilla_params,
    inductive_energy_ghz,
    charging_energy_ghz,
)

def _peak_spacing_from_pairwise_diffs(energies, min_diff=1.0, bin_width=0.05):
    energies = np.array(sorted(energies))
    diffs = []
    for i in range(len(energies)):
        for j in range(i + 1, len(energies)):
            d = energies[j] - energies[i]
            if d >= min_diff:
                diffs.append(float(d))
    diffs = np.array(diffs)
    if diffs.size == 0:
        raise ValueError("Not enough diffs to estimate a spacing.")
    dmin, dmax = diffs.min(), diffs.max()
    nbins = max(10, int(np.ceil((dmax - dmin) / bin_width)))
    hist, edges = np.histogram(diffs, bins=nbins, range=(dmin, dmax))
    k = int(np.argmax(hist))
    return 0.5 * (edges[k] + edges[k + 1])


def _nearest_int_multiple_residual(E, omega):
    m = int(np.round(E / omega))
    return abs(E - m * omega), m


def _identify_omega_r_from_hint(E, omega_r_hint):
    """Find the energy level closest to omega_r_hint; that level is |0,0,1>, so omega_r = E[that]."""
    E = np.asarray(E)
    k = int(np.argmin(np.abs(E - omega_r_hint)))
    return float(E[k]), k


def _match_nearest_unused_level(E, target, used_indices=None):
    E = np.asarray(E)
    if used_indices is None:
        used_indices = set()
    order = np.argsort(np.abs(E - target))
    for idx in order:
        idx = int(idx)
        if idx not in used_indices:
            return idx
    raise ValueError("No unused energy levels available for matching.")


def fit_effective_params_3mode(
    evals_rel,
    tol_nonmultiple=0.20,
    omega_r_known=None,
    omega_r_hint=None,
    omega_q_hint=None,
    omega_a_hint=None,
):
    """
    Fit the effective 3-mode Hamiltonian parameters from a spectrum.

    H = omega_q*nq + alpha_q/2 * nq*(nq-1)
      + omega_a*na + alpha_a/2 * na*(na-1)
      + omega_r*nr
      + chi_qa*nq*na + chi_ar*na*nr + chi_qr*nq*nr

    Physical frequency hints (GHz) from the design LC formulas:
        omega_r_hint  ~  1/(2*pi*sqrt(L_r * C_r))
        omega_q_hint  ~  1/(2*pi*sqrt(L_J_qubit * C_shunt))
        omega_a_hint  ~  1/(2*pi*sqrt(L_J_ancilla * C_ancilla))

    When a hint is supplied, the nearest spectral level to that hint is used
    as the mode frequency.  If omega_a_hint falls above the computed spectrum
    (ancilla is a high-frequency mode), omega_a is taken directly from the
    hint and alpha_a / chi_qa / chi_ar are set to 0 (not observable).

    Returns dict with keys:
        omega_q, omega_a, omega_r, alpha_q, alpha_a,
        chi_qa, chi_ar, chi_qr   (all in GHz)
    """
    E = np.array(sorted(evals_rel))
    if E[0] != 0.0:
        E = E - E[0]
    E_max = float(E[-1])

    # ── Step 1: omega_r ───────────────────────────────────────────────────────
    if omega_r_known is not None:
        omega_r = float(omega_r_known)
    elif omega_r_hint is not None:
        omega_r, _ = _identify_omega_r_from_hint(E, float(omega_r_hint))
    else:
        omega_r = _peak_spacing_from_pairwise_diffs(
            E[: min(len(E), 20)], min_diff=1.0, bin_width=0.02
        )

    # ── Step 2: omega_q ───────────────────────────────────────────────────────
    # Prefer the physical hint (nearest spectral level to the bare LC estimate).
    if omega_q_hint is not None:
        omega_q, _ = _identify_omega_r_from_hint(E, float(omega_q_hint))
    else:
        # Fallback: first spectral level that isn't a multiple of omega_r
        omega_q = None
        for Ek in E[1:]:
            resid, _ = _nearest_int_multiple_residual(Ek, omega_r)
            if resid > tol_nonmultiple:
                omega_q = float(Ek)
                break
        if omega_q is None:
            omega_q = float(E[1])

    # ── Step 3: omega_a ───────────────────────────────────────────────────────
    # If the physical hint is above the computed spectrum, use it directly —
    # the ancilla is a high-frequency mode and its photon states don't appear
    # in the low-energy sector.  alpha_a and the cross-Kerr terms involving the
    # ancilla are then not observable and are set to 0.
    ancilla_above_spectrum = False
    if omega_a_hint is not None:
        if float(omega_a_hint) > E_max:
            omega_a = float(omega_a_hint)
            ancilla_above_spectrum = True
        else:
            omega_a, _ = _identify_omega_r_from_hint(E, float(omega_a_hint))
    else:
        # Fallback: second non-resonator level
        omega_a = None
        count = 0
        for Ek in E[1:]:
            resid, _ = _nearest_int_multiple_residual(Ek, omega_r)
            if resid > tol_nonmultiple and abs(Ek - omega_q) > tol_nonmultiple:
                omega_a = float(Ek)
                break
        if omega_a is None:
            omega_a = float(E[3]) if len(E) > 3 else float(E[2])

    # ── Step 4: alpha_q from |2,0,0> ─────────────────────────────────────────
    # Report alpha_q as the positive anharmonicity magnitude:
    #   alpha_q = f_01 - f_12 = 2*omega_q - E|2,0,0>  ~ E_C
    target_2q = 2.0 * omega_q
    idx_2q = int(np.argmin(np.abs(E - target_2q)))
    alpha_q = float(target_2q - E[idx_2q])

    # ── Steps 5-7: ancilla-involving quantities ───────────────────────────────
    if ancilla_above_spectrum:
        # |0,2,0>, |1,1,0>, |0,1,1> are all above the spectrum → not observable
        alpha_a = 0.0
        chi_qa  = 0.0
        chi_ar  = 0.0
    else:
        target_2a = 2.0 * omega_a
        idx_2a    = int(np.argmin(np.abs(E - target_2a)))
        alpha_a   = float(target_2a - E[idx_2a])

        target_qa = omega_q + omega_a
        idx_qa    = int(np.argmin(np.abs(E - target_qa)))
        chi_qa    = float(E[idx_qa] - target_qa)

        target_ar = omega_a + omega_r
        idx_ar    = int(np.argmin(np.abs(E - target_ar)))
        chi_ar    = float(E[idx_ar] - target_ar)

    # ── Step 8: chi_qr from |1,0,1> ──────────────────────────────────────────
    target_qr = omega_q + omega_r
    idx_qr    = int(np.argmin(np.abs(E - target_qr)))
    chi_qr    = float(E[idx_qr] - target_qr)

    return dict(
        omega_r=omega_r,
        omega_q=omega_q,
        omega_a=omega_a,
        alpha_q=alpha_q,
        alpha_a=alpha_a,
        chi_qa=chi_qa,
        chi_ar=chi_ar,
        chi_qr=chi_qr,
        _ancilla_above_spectrum=ancilla_above_spectrum,
    )


def predicted_energy_3mode(nq, na, nr, p, include_chi=True):
    """
    Predicted energy of state |nq, na, nr> from the 3-mode Hamiltonian:

    H = omega_q*nq - alpha_q/2 * nq*(nq-1)
      + omega_a*na - alpha_a/2 * na*(na-1)
      + omega_r*nr
      + chi_qa*nq*na + chi_ar*na*nr + chi_qr*nq*nr
    """
    omega_q = p["omega_q"]
    omega_a = p["omega_a"]
    omega_r = p["omega_r"]
    alpha_q = p["alpha_q"]
    alpha_a = p["alpha_a"]
    chi_qa = p["chi_qa"] if include_chi else 0.0
    chi_ar = p["chi_ar"] if include_chi else 0.0
    chi_qr = p["chi_qr"] if include_chi else 0.0
    return (
        omega_q * nq
        - 0.5 * alpha_q * nq * (nq - 1)
        + omega_a * na
        - 0.5 * alpha_a * na * (na - 1)
        + omega_r * nr
        + chi_qa * nq * na
        + chi_ar * na * nr
        + chi_qr * nq * nr
    )


def assign_labels_3mode(evals_rel, p, nq_max=4, na_max=4, nr_max=8, include_chi=True):
    E = np.array(sorted(evals_rel))
    E = E - E[0]

    cand = []
    for nq in range(nq_max + 1):
        for na in range(na_max + 1):
            for nr in range(nr_max + 1):
                Ep = predicted_energy_3mode(nq, na, nr, p, include_chi=include_chi)
                cand.append((Ep, nq, na, nr))
    cand.sort(key=lambda t: t[0])

    # Anchor the single-mode ladders first using integer multiples of the
    # identified fundamentals: f_q, f_a, f_r, then 2*f_q, 2*f_a, 2*f_r, etc.
    e_max = float(E[-1])
    anchor_specs = [(0.0, 0, 0, 0)]
    for nq in range(1, nq_max + 1):
        target = nq * p["omega_q"]
        if target <= e_max + 1.0:
            anchor_specs.append((target, nq, 0, 0))
    for na in range(1, na_max + 1):
        target = na * p["omega_a"]
        if target <= e_max + 1.0:
            anchor_specs.append((target, 0, na, 0))
    for nr in range(1, nr_max + 1):
        target = nr * p["omega_r"]
        if target <= e_max + 1.0:
            anchor_specs.append((target, 0, 0, nr))
    anchor_specs.sort(key=lambda t: t[0])

    used = set()
    used_energy_indices = set()
    anchored_by_index = {}
    for target, nq, na, nr in anchor_specs:
        idx = _match_nearest_unused_level(E, target, used_energy_indices)
        anchored_by_index[idx] = dict(
            k=idx,
            E=E[idx],
            nq=nq,
            na=na,
            nr=nr,
            E_pred=predicted_energy_3mode(nq, na, nr, p, include_chi=include_chi),
            resid=abs(E[idx] - target),
        )
        used.add((nq, na, nr))
        used_energy_indices.add(idx)

    out = []
    for k, Ek in enumerate(E):
        if k in anchored_by_index:
            out.append(anchored_by_index[k])
            continue

        best = None
        best_score = 1e99

        for Ep, nq, na, nr in cand:
            if (nq, na, nr) in used:
                continue
            if abs(Ep - Ek) > 1.0:
                continue
            resid = abs(Ep - Ek)
            penalty = 1e-3 * (nq + na + nr)
            score = resid + penalty
            if score < best_score:
                best_score = score
                best = (nq, na, nr, Ep, resid)

        if best is None:
            # No candidate within 1 GHz — fall back to global nearest
            for Ep, nq, na, nr in cand:
                if (nq, na, nr) in used:
                    continue
                resid = abs(Ep - Ek)
                penalty = 1e-3 * (nq + na + nr)
                score = resid + penalty
                if score < best_score:
                    best_score = score
                    best = (nq, na, nr, Ep, resid)

        nq, na, nr, Ep, resid = best
        used.add((nq, na, nr))
        out.append(dict(k=k, E=Ek, nq=nq, na=na, nr=nr, E_pred=Ep, resid=resid))

    return out


# ── Physical LC frequency hints from design_graph.txt ─────────────────────────
# omega_r = 1/sqrt(L_r * C_r)           (resonator inductor and capacitor)
# omega_q = 1/sqrt(L_J_qubit * C_shunt)  (SQUID junction + shunt cap)
# omega_a = 1/sqrt(L_J_ancilla * C_ancilla) (ancilla junction + ancilla capacitance)

L_J_q, C_shunt_q     = get_qubit_params()
L_J_a, C_ancilla_a   = get_ancilla_params()

print(f"L_J_q: {L_J_q}, C_shunt_q: {C_shunt_q}")
print(f"L_J_a: {L_J_a}, C_ancilla_a: {C_ancilla_a}")

OMEGA_R_HINT = f_res_bare
OMEGA_Q_HINT = 1.0 / (np.sqrt(L_J_q * C_shunt_q)    * 2 * np.pi * 1e9)
OMEGA_A_HINT = 1.0 / (np.sqrt(L_J_a * C_ancilla_a) * 2 * np.pi * 1e9)

print("\n[Physical LC frequency hints from design_graph.txt]")
print(f"  omega_r_hint = 1/sqrt(L_r*C_r)           = {OMEGA_R_HINT:.4f} GHz")
print(f"  omega_q_hint = 1/sqrt(L_J_q*C_shunt)     = {OMEGA_Q_HINT:.4f} GHz"
      f"  (L_J={L_J_q*1e9:.1f} nH, C={C_shunt_q*1e15:.0f} fF)")
print(f"  omega_a_hint = 1/sqrt(L_J_a*C_ancilla)   = {OMEGA_A_HINT:.4f} GHz"
      f"  (L_J={L_J_a*1e9:.1f} nH, C={C_ancilla_a*1e15:.0f} fF)")

params = fit_effective_params_3mode(
    evals_rel,
    omega_r_hint=OMEGA_R_HINT,
    omega_q_hint=OMEGA_Q_HINT,
    omega_a_hint=OMEGA_A_HINT,
)
assign_nochi = assign_labels_3mode(evals_rel, params, include_chi=False)
assign_chi   = assign_labels_3mode(evals_rel, params, include_chi=True)

above = params.get("_ancilla_above_spectrum", False)
print("\n[Fitted effective 3-mode parameters] (GHz)")
print(f"  omega_r = {params['omega_r']:.6f}  (hint {OMEGA_R_HINT:.3f} GHz)")
print(f"  omega_q = {params['omega_q']:.6f}  (hint {OMEGA_Q_HINT:.3f} GHz, nearest level in spectrum)")
if above:
    print(f"  omega_a = {params['omega_a']:.4f}  (from LC hint — ancilla above computed spectrum)")
    print(f"  alpha_a : not observable from low-energy spectrum")
    print(f"  chi_qa  : not observable from low-energy spectrum")
    print(f"  chi_ar  : not observable from low-energy spectrum")
else:
    print(f"  omega_a = {params['omega_a']:.6f}  (hint {OMEGA_A_HINT:.3f} GHz, nearest level in spectrum)")
    print(f"  alpha_a = {params['alpha_a']:.6f}  (= f_a01 - f_a12)")
    print(f"  chi_qa  = {params['chi_qa']:.6f}  (= E|1,1,0> - E|0,1,0> - E|1,0,0>)")
    print(f"  chi_ar  = {params['chi_ar']:.6f}  (= E|0,1,1> - E|0,0,1> - E|0,1,0>)")
print(f"  alpha_q = {params['alpha_q']:.6f}  (= f_q01 - f_q12)")
print(f"  chi_qr  = {params['chi_qr']:.6f}  (= E|1,0,1> - E|0,0,1> - E|1,0,0>)")


def _pretty_print(assignments, title):
    print(f"\n{title}")
    print("  idx    E_obs(GHz)   label           E_pred(GHz)     |Δ|(MHz)")
    for a in assignments:
        print(
            f"  |{a['k']:2d}>  {a['E']:11.6f}   "
            f"|{a['nq']},{a['na']},{a['nr']}>   "
            f"{a['E_pred']:11.6f}    {1e3 * a['resid']:8.2f}"
        )

_pretty_print(assign_nochi, "[Labeling using Kerr + harmonic only (no chi)]")
_pretty_print(assign_chi,   "[Labeling using full 3-mode Hamiltonian]")

def energy_of_label(assignments, nq, na, nr):
    for a in assignments:
        if a["nq"] == nq and a["na"] == na and a["nr"] == nr:
            return a["E"]
    return None


E000  = energy_of_label(assign_chi, 0, 0, 0)
Eq100 = energy_of_label(assign_chi, 1, 0, 0)
Ea010 = energy_of_label(assign_chi, 0, 1, 0)
Er001 = energy_of_label(assign_chi, 0, 0, 1)

print("\n[Fundamentals inferred from labels] (GHz)")
if Eq100 is not None:
    print(f"  f_q01 ≈ E|1,0,0> - E|0,0,0> = {Eq100 - E000:.6f}")
if Ea010 is not None:
    print(f"  f_a01 ≈ E|0,1,0> - E|0,0,0> = {Ea010 - E000:.6f}")
if Er001 is not None:
    print(f"  f_r01 ≈ E|0,0,1> - E|0,0,0> = {Er001 - E000:.6f}")

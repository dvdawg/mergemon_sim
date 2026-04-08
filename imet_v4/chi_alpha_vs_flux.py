import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scqubits as scq
from scipy.optimize import linear_sum_assignment, minimize

from circuit_from_design import (
    apply_recommended_cutoffs,
    build_circuit as build_circuit_from_design,
    get_resonator_params,
    get_qubit_params,
    get_ancilla_params,
    inductive_energy_ghz,
    charging_energy_ghz,
)

# Design parameters read directly from design_graph.txt
L_r, C_r           = get_resonator_params()   # resonator L and C
L_J_q, C_shunt_q   = get_qubit_params()       # SQUID JJ + shunt cap
L_J_a, C_ancilla_a = get_ancilla_params()     # ancilla JJ + ancilla capacitance

# Bare LC frequency hints (GHz):  omega = 1/sqrt(L*C)
OMEGA_R_GHZ = 1.0 / (np.sqrt(L_r   * C_r)           * 2 * np.pi * 1e9)
OMEGA_Q_GHZ = 1.0 / (np.sqrt(L_J_q * C_shunt_q)     * 2 * np.pi * 1e9)
OMEGA_A_GHZ = 1.0 / (np.sqrt(L_J_a * C_ancilla_a)   * 2 * np.pi * 1e9)

print(f"Resonator: L_r={L_r*1e9:.2f} nH, C_r={C_r*1e12:.1f} pF  →  f_r_hint={OMEGA_R_GHZ:.4f} GHz")
print(f"Qubit:     L_J={L_J_q*1e9:.1f} nH, C={C_shunt_q*1e15:.0f} fF      →  f_q_hint={OMEGA_Q_GHZ:.4f} GHz")
print(f"Ancilla:   L_J={L_J_a*1e9:.1f} nH, C={C_ancilla_a*1e15:.0f} fF      →  f_a_hint={OMEGA_A_GHZ:.4f} GHz")

E_J_data = inductive_energy_ghz(L_J_q)    # single SQUID junction E_J (GHz)
E_C_eff  = charging_energy_ghz(C_shunt_q) # qubit charging energy (shunt cap dominates)
d_asym   = 0.0   # symmetric SQUID

NQ_MAX = 4
NA_MAX = 3
NR_MAX = 6

def E_J_squid(phi_ext):
    phi_ext = np.asarray(phi_ext, dtype=float)
    arg     = np.pi * phi_ext
    cos_val = np.cos(arg)
    with np.errstate(divide="ignore", invalid="ignore"):
        tan_val = np.tan(arg)
        E_J = (2 * E_J_data) * np.abs(cos_val) * np.sqrt(
            1.0 + d_asym**2 * tan_val**2
        )
    return np.where(np.isfinite(E_J), E_J, 0.0)


def transmon_energy_levels(E_J, E_C, n_max=5):
    if E_J <= 0:
        n = np.arange(n_max + 1, dtype=float)
        return n * E_C
    n = np.arange(n_max + 1, dtype=float)
    E_n = (
        np.sqrt(8.0 * E_J * E_C) * (n + 0.5)
        - (E_C / 12.0) * (6 * n**2 + 6 * n + 3)
        - E_J
    )
    return E_n - E_n[0]


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


def _identify_omega_from_hint(E, omega_hint):
    E = np.asarray(E)
    k = int(np.argmin(np.abs(E - omega_hint)))
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


def predict_3mode(phi_ext, analytical_params):
    """
    Predicted energies for states |nq, na, nr> using the 3-mode Hamiltonian:

    H = omega_q*nq - alpha_q/2*nq*(nq-1)
      + omega_a*na - alpha_a/2*na*(na-1)
      + omega_r*nr
      + chi_qa*nq*na + chi_ar*na*nr + chi_qr*nq*nr

    The qubit levels come from the transmon formula (flux-tunable via SQUID).
    The ancilla frequency omega_a and alpha_a are fitted from the spectrum.
    omega_r is the bare LC resonator frequency.
    """
    E_J = float(E_J_squid(phi_ext))
    eq  = transmon_energy_levels(E_J, E_C_eff, n_max=NQ_MAX)

    omega_a = analytical_params["omega_a"]
    alpha_a = analytical_params["alpha_a"]
    chi_qa  = analytical_params["chi_qa"]
    chi_ar  = analytical_params["chi_ar"]
    chi_qr  = analytical_params["chi_qr"]

    preds = {}
    for nq in range(NQ_MAX + 1):
        for na in range(NA_MAX + 1):
            Ea = omega_a * na - 0.5 * alpha_a * na * (na - 1)
            for nr in range(NR_MAX + 1):
                E = (
                    eq[nq] + Ea + nr * OMEGA_R_GHZ
                    + chi_qa * nq * na
                    + chi_ar * na * nr
                    + chi_qr * nq * nr
                )
                preds[(nq, na, nr)] = E
    return preds


def fit_effective_params_3mode(
    evals_rel,
    tol_nonmultiple=0.20,
    omega_r_known=None,
    omega_r_hint=None,
    omega_q_hint=None,
    omega_a_hint=None,
):
    E = np.array(sorted(evals_rel))
    if E[0] != 0.0:
        E = E - E[0]
    E_max = float(E[-1])

    if omega_r_known is not None:
        omega_r = float(omega_r_known)
    elif omega_r_hint is not None:
        omega_r, _ = _identify_omega_from_hint(E, float(omega_r_hint))
    else:
        omega_r = _peak_spacing_from_pairwise_diffs(
            E[: min(len(E), 20)], min_diff=1.0, bin_width=0.02
        )

    if omega_q_hint is not None:
        omega_q, _ = _identify_omega_from_hint(E, float(omega_q_hint))
    else:
        omega_q = None
        for Ek in E[1:]:
            resid, _ = _nearest_int_multiple_residual(Ek, omega_r)
            if resid > tol_nonmultiple:
                omega_q = float(Ek)
                break
        if omega_q is None:
            omega_q = float(E[1])

    ancilla_above_spectrum = False
    if omega_a_hint is not None:
        if float(omega_a_hint) > E_max:
            omega_a = float(omega_a_hint)
            ancilla_above_spectrum = True
        else:
            omega_a, _ = _identify_omega_from_hint(E, float(omega_a_hint))
    else:
        omega_a = None
        for Ek in E[1:]:
            resid, _ = _nearest_int_multiple_residual(Ek, omega_r)
            if resid > tol_nonmultiple and abs(Ek - omega_q) > tol_nonmultiple:
                omega_a = float(Ek)
                break
        if omega_a is None:
            omega_a = float(E[3]) if len(E) > 3 else float(E[2])

    target_2q = 2.0 * omega_q
    idx_2q = int(np.argmin(np.abs(E - target_2q)))
    alpha_q = float(target_2q - E[idx_2q])

    if ancilla_above_spectrum:
        alpha_a = 0.0
        chi_qa = 0.0
        chi_ar = 0.0
    else:
        target_2a = 2.0 * omega_a
        idx_2a = int(np.argmin(np.abs(E - target_2a)))
        alpha_a = float(target_2a - E[idx_2a])

        target_qa = omega_q + omega_a
        idx_qa = int(np.argmin(np.abs(E - target_qa)))
        chi_qa = float(E[idx_qa] - target_qa)

        target_ar = omega_a + omega_r
        idx_ar = int(np.argmin(np.abs(E - target_ar)))
        chi_ar = float(E[idx_ar] - target_ar)

    target_qr = omega_q + omega_r
    idx_qr = int(np.argmin(np.abs(E - target_qr)))
    chi_qr = float(E[idx_qr] - target_qr)

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


def predicted_energy_from_fit(nq, na, nr, p, include_chi=True):
    chi_qa = p["chi_qa"] if include_chi else 0.0
    chi_ar = p["chi_ar"] if include_chi else 0.0
    chi_qr = p["chi_qr"] if include_chi else 0.0
    return (
        p["omega_q"] * nq
        - 0.5 * p["alpha_q"] * nq * (nq - 1)
        + p["omega_a"] * na
        - 0.5 * p["alpha_a"] * na * (na - 1)
        + p["omega_r"] * nr
        + chi_qa * nq * na
        + chi_ar * na * nr
        + chi_qr * nq * nr
    )


def assign_labels_3mode(evals_rel, p, nq_max=NQ_MAX, na_max=NA_MAX, nr_max=NR_MAX, include_chi=True):
    E = np.array(sorted(evals_rel))
    E = E - E[0]

    cand = []
    for nq in range(nq_max + 1):
        for na in range(na_max + 1):
            for nr in range(nr_max + 1):
                Ep = predicted_energy_from_fit(nq, na, nr, p, include_chi=include_chi)
                cand.append((Ep, nq, na, nr))
    cand.sort(key=lambda t: t[0])

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
            E_pred=predicted_energy_from_fit(nq, na, nr, p, include_chi=include_chi),
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


def _match_branches_step(
    prev_evals,
    curr_evals,
    prev_evecs,
    curr_evecs,
    delta_phi,
    prev_prev_evals=None,
    overlap_weight=4.0,
    energy_weight=0.7,
    slope_weight=1.6,
    energy_scale=0.05,
    slope_scale=0.02,
):
    overlap = np.abs(curr_evecs.conj().T @ prev_evecs) ** 2
    cost = overlap_weight * (1.0 - overlap)
    cost += energy_weight * (np.abs(curr_evals[:, None] - prev_evals[None, :]) / energy_scale)

    if prev_prev_evals is not None:
        prev_slope = (prev_evals - prev_prev_evals) / delta_phi
        predicted = prev_evals + prev_slope * delta_phi
        cost += slope_weight * (np.abs(curr_evals[:, None] - predicted[None, :]) / slope_scale)

    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.empty_like(col_ind)
    perm[col_ind] = row_ind
    return perm


def track_eigenbranches(raw_evals, all_evecs, phi_vals, start_idx):
    n_flux, _ = raw_evals.shape
    tracked_evals = np.zeros_like(raw_evals)
    tracked_evecs = [None] * n_flux

    tracked_evals[start_idx] = raw_evals[start_idx]
    tracked_evecs[start_idx] = all_evecs[start_idx]

    for i in range(start_idx + 1, n_flux):
        prev_prev_evals = tracked_evals[i - 2] if i - 2 >= start_idx else None
        delta_phi = float(phi_vals[i] - phi_vals[i - 1])
        perm = _match_branches_step(
            tracked_evals[i - 1], raw_evals[i], tracked_evecs[i - 1], all_evecs[i],
            delta_phi, prev_prev_evals=prev_prev_evals
        )
        tracked_evals[i] = raw_evals[i][perm]
        tracked_evecs[i] = all_evecs[i][:, perm]

    for i in range(start_idx - 1, -1, -1):
        prev_prev_evals = tracked_evals[i + 2] if i + 2 <= start_idx else None
        delta_phi = float(phi_vals[i] - phi_vals[i + 1])
        perm = _match_branches_step(
            tracked_evals[i + 1], raw_evals[i], tracked_evecs[i + 1], all_evecs[i],
            delta_phi, prev_prev_evals=prev_prev_evals
        )
        tracked_evals[i] = raw_evals[i][perm]
        tracked_evecs[i] = all_evecs[i][:, perm]

    return tracked_evals, tracked_evecs


def fit_analytical_params_3mode(evals_sweet, phi_sweet=0.0, omega_a_hint=None):
    """
    Fit omega_a, alpha_a, chi_qa, chi_ar, chi_qr from the sweet-spot spectrum
    by minimising the Hungarian-assignment cost against the 3-mode prediction.

    omega_a_hint (GHz): physical LC estimate 1/(2*pi*sqrt(L_J_a * C_ancilla)).
    Used as the initial guess.  If omega_a_hint is above the spectrum, omega_a
    is fixed at the hint and only the chi values are optimised.

    Returns analytical_params dict.
    """
    E = np.asarray(evals_sweet, dtype=float)
    E = E - E[0]
    E_max = float(E[-1])

    # Determine initial guess for omega_a
    if omega_a_hint is not None:
        omega_a_guess = float(omega_a_hint)
    else:
        # Scan spectrum for levels not explained by qubit+resonator
        E_J_sw = float(E_J_squid(phi_sweet))
        eq_sw  = transmon_energy_levels(E_J_sw, E_C_eff, n_max=NQ_MAX)
        omega_a_guess = OMEGA_R_GHZ  # fallback
        for Ek in E[1:]:
            is_qr = any(
                abs(Ek - (eq_sw[nq] + nr * OMEGA_R_GHZ)) < 0.4
                for nq in range(NQ_MAX + 1)
                for nr in range(NR_MAX + 1)
            )
            if not is_qr:
                omega_a_guess = float(Ek)
                break

    ancilla_above = (omega_a_hint is not None and float(omega_a_hint) > E_max)

    x0 = [omega_a_guess, 0.0, 0.0, 0.0, 0.0]  # omega_a, alpha_a, chi_qa, chi_ar, chi_qr

    def residual(x):
        omega_a, alpha_a, chi_qa, chi_ar, chi_qr = x
        p = dict(omega_a=omega_a, alpha_a=alpha_a,
                 chi_qa=chi_qa, chi_ar=chi_ar, chi_qr=chi_qr)
        preds = predict_3mode(phi_sweet, p)
        cands = sorted(preds.items(), key=lambda kv: kv[1])
        N = min(len(E), len(cands))
        cost = np.array(
            [[abs(E[i] - cands[j][1]) for j in range(N)] for i in range(N)]
        )
        r, c = linear_sum_assignment(cost)
        return float(cost[r, c].sum())

    result = minimize(
        residual, x0,
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 5000},
    )
    omega_a, alpha_a, chi_qa, chi_ar, chi_qr = result.x
    return dict(omega_a=omega_a, alpha_a=alpha_a,
                chi_qa=chi_qa, chi_ar=chi_ar, chi_qr=chi_qr)


print("Building circuit from design_graph.txt...")
circ, _ = build_circuit_from_design()
apply_recommended_cutoffs(circ, periodic_cutoff=8, extended_cutoff=12)

flux_syms  = circ.external_fluxes
flux_attrs = [str(s) for s in flux_syms]

if not flux_attrs:
    sys.exit("No external flux variables found - check circuit topology.")
print(f"External flux variables: {flux_attrs}")

for a in flux_attrs:
    setattr(circ, a, 0.0)

ev_base, _ = circ.eigensys(evals_count=5)
E1_base    = (ev_base - ev_base[0])[1]
squid_attr = flux_attrs[0]

if len(flux_attrs) > 1:
    best_shift = -1.0
    for a in flux_attrs:
        for b in flux_attrs:
            setattr(circ, b, 0.0)
        setattr(circ, a, 0.5)
        ev_t, _ = circ.eigensys(evals_count=5)
        shift = abs((ev_t - ev_t[0])[1] - E1_base)
        print(f"{a} = 0.5 Phi_0  ->  |Delta f_01| = {shift:.4f} GHz")
        if shift > best_shift:
            best_shift, squid_attr = shift, a
    for a in flux_attrs:
        setattr(circ, a, 0.0)

print(f"SQUID flux variable: '{squid_attr}'")
for a in flux_attrs:
    if a != squid_attr:
        setattr(circ, a, 0.0)

# ── Flux sweep ────────────────────────────────────────────────────────────────
N_LEVELS = 16
N_FLUX   = 101
phi_vals = np.linspace(-0.5, 0.5, N_FLUX)
mid_idx  = N_FLUX // 2

raw_evals = np.zeros((N_FLUX, N_LEVELS))
all_evecs = []

print(f"\nSweeping {N_FLUX} flux points, {N_LEVELS} eigenvalues each...")
for i, phi in enumerate(phi_vals):
    setattr(circ, squid_attr, phi)
    ev, evec = circ.eigensys(evals_count=N_LEVELS)
    raw_evals[i] = ev - ev[0]
    evec = np.array(evec)
    if evec.ndim == 1:
        evec = evec[:, np.newaxis]
    if evec.shape[0] < evec.shape[1]:
        evec = evec.T
    all_evecs.append(evec)
    if (i + 1) % 20 == 0 or i == 0:
        print(f"  {i + 1:3d} / {N_FLUX}")

print("\nTracking eigenvalues...")
all_evals, all_evecs_tracked = track_eigenbranches(raw_evals, all_evecs, phi_vals, mid_idx)

print("Tracking complete.")

# ── Identify sweet-spot levels with the combined logic ────────────────────────
print("\nIdentifying sweet-spot levels using combined identification logic...")
sweet_params = fit_effective_params_3mode(
    all_evals[mid_idx],
    omega_r_hint=OMEGA_R_GHZ,
    omega_q_hint=OMEGA_Q_GHZ,
    omega_a_hint=OMEGA_A_GHZ,
)
sweet_assignments = assign_labels_3mode(all_evals[mid_idx], sweet_params, include_chi=True)
sweet_labels = {a["k"]: (a["nq"], a["na"], a["nr"]) for a in sweet_assignments}

# ── Optional analytical fit for comparison/printing ───────────────────────────
print("\nFitting 3-mode analytical parameters at sweet spot...")
analytical_params = fit_analytical_params_3mode(
    all_evals[mid_idx], phi_sweet=phi_vals[mid_idx], omega_a_hint=OMEGA_A_GHZ
)
print(f"  omega_a = {analytical_params['omega_a']:.4f} GHz")
print(f"  alpha_a = {analytical_params['alpha_a'] * 1e3:.1f} MHz")
print(f"  chi_qr  = {analytical_params['chi_qr'] * 1e3:.2f} MHz")

# ── Attach one label to each tracked branch ───────────────────────────────────
print("Attaching sweet-spot labels to tracked branches...")
tracked_branch_labels = {k: sweet_labels.get(k, (-1, -1, -1)) for k in range(N_LEVELS)}

# ── Extract chi and alpha vs flux ─────────────────────────────────────────────
alpha_q_vals = np.full(N_FLUX, np.nan)
alpha_r_vals = np.full(N_FLUX, np.nan)
chi_qr_vals  = np.full(N_FLUX, np.nan)

for i in range(N_FLUX):
    qn_to_E = {
        label: all_evals[i, k]
        for k, label in tracked_branch_labels.items()
        if label != (-1, -1, -1)
    }

    E_000 = qn_to_E.get((0, 0, 0))
    E_100 = qn_to_E.get((1, 0, 0))
    E_200 = qn_to_E.get((2, 0, 0))
    E_001 = qn_to_E.get((0, 0, 1))
    E_002 = qn_to_E.get((0, 0, 2))
    E_101 = qn_to_E.get((1, 0, 1))

    # Positive anharmonicity convention: alpha = f01 - f12
    if all(x is not None for x in [E_000, E_100, E_200]):
        alpha_q_vals[i] = 2.0 * E_100 - E_200 - E_000

    if all(x is not None for x in [E_000, E_001, E_002]):
        alpha_r_vals[i] = 2.0 * E_001 - E_002 - E_000

    if all(x is not None for x in [E_000, E_100, E_001, E_101]):
        chi_qr_vals[i] = (E_101 - E_001) - (E_100 - E_000)

alpha_q_mhz = alpha_q_vals * 1e3
alpha_r_mhz = alpha_r_vals * 1e3
chi_qr_mhz  = chi_qr_vals  * 1e3

print(f"\n--- Values at sweet spot (Phi = {phi_vals[mid_idx]:.3f} Phi_0) ---")
for name, arr in [
    ("alpha_q", alpha_q_mhz),
    ("alpha_r", alpha_r_mhz),
    ("chi_qr",  chi_qr_mhz),
]:
    v = arr[mid_idx]
    s = f"{v:.2f}" if not np.isnan(v) else "N/A"
    print(f"  {name} / 2pi = {s} MHz")

def _finite_bounds(arrays, pad_frac=0.08, min_span=1.0):
    finite = np.concatenate([np.asarray(a, dtype=float)[np.isfinite(a)] for a in arrays])
    if finite.size == 0:
        return -0.5, 0.5
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    span = max(vmax - vmin, min_span)
    pad = pad_frac * span
    return vmin - pad, vmax + pad

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(phi_vals, chi_qr_mhz, color="C0", linewidth=1.8, label=r"$\chi_{qr}$")
ax1.set_ylabel(r"$\chi\,/\,2\pi$ (MHz)", fontsize=12)
ax1.set_title("Qubit-Resonator Dispersive Shift vs External Flux", fontsize=13)
ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax1.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.25)
ax1.set_xlim(float(phi_vals[0]), float(phi_vals[-1]))
ax1.set_ylim(*_finite_bounds([chi_qr_mhz], pad_frac=0.10, min_span=0.2))

ax2.plot(phi_vals, alpha_q_mhz, color="C1", linewidth=1.6, label=r"$\alpha_q$")
ax2.plot(phi_vals, alpha_r_mhz, color="C4", linewidth=1.6, label=r"$\alpha_r$")
ax2.set_xlabel(r"External flux  $\Phi_\mathrm{ext}\,/\,\Phi_0$", fontsize=12)
ax2.set_ylabel(r"$\alpha\,/\,2\pi$ (MHz)", fontsize=12)
ax2.set_title("Qubit and Resonator Anharmonicity vs External Flux", fontsize=13)
ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax2.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.25)
ax2.set_xlim(float(phi_vals[0]), float(phi_vals[-1]))
ax2.set_ylim(*_finite_bounds([alpha_q_mhz, alpha_r_mhz], pad_frac=0.10, min_span=0.2))

fig.tight_layout()

os.makedirs("plot_output", exist_ok=True)
out_path = "plot_output/chi_alpha_vs_flux_v2.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
plt.show()

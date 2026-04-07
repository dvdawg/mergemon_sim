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

AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
TITLE_FONTSIZE      = 14
LEGEND_FONTSIZE     = 10
SUPTITLE_FONTSIZE   = 18

# Design parameters read directly from design_graph.txt
L_r, C_r            = get_resonator_params()
L_J_q, C_shunt_q    = get_qubit_params()
L_J_a, C_ancilla_a = get_ancilla_params()

# Bare LC frequency hints (GHz):  omega = 1/sqrt(L*C)
OMEGA_R_GHZ = 1.0 / (np.sqrt(L_r   * C_r)           * 2 * np.pi * 1e9)
OMEGA_Q_GHZ = 1.0 / (np.sqrt(L_J_q * C_shunt_q)     * 2 * np.pi * 1e9)
OMEGA_A_GHZ = 1.0 / (np.sqrt(L_J_a * C_ancilla_a)   * 2 * np.pi * 1e9)

print(f"Resonator: L_r={L_r*1e9:.2f} nH, C_r={C_r*1e12:.1f} pF  →  f_r_hint={OMEGA_R_GHZ:.4f} GHz")
print(f"Qubit:     L_J={L_J_q*1e9:.1f} nH, C={C_shunt_q*1e15:.0f} fF      →  f_q_hint={OMEGA_Q_GHZ:.4f} GHz")
print(f"Ancilla:   L_J={L_J_a*1e9:.1f} nH, C={C_ancilla_a*1e15:.0f} fF      →  f_a_hint={OMEGA_A_GHZ:.4f} GHz")

E_J_data = inductive_energy_ghz(L_J_q)    # single SQUID junction E_J (GHz)
E_C_eff  = charging_energy_ghz(C_shunt_q) # qubit charging energy
d_asym   = 0.0

NQ_MAX = 4
NA_MAX = 3
NR_MAX = 6


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


def predict_3mode_from_sweet_anchor(phi_ext, sweet_energy_by_label, phi_ref=0.0):
    """
    Predict the flux evolution by anchoring every labelled state to its
    *numerical* sweet-spot energy and only applying the SQUID-induced qubit
    shift away from that point.

    This keeps the comparison panel aligned with the trusted sweet-spot
    diagonalization/identification while still showing the intended analytical
    flux trend.
    """
    eq_ref = transmon_energy_levels(float(E_J_squid(phi_ref)), E_C_eff, n_max=NQ_MAX)
    eq_phi = transmon_energy_levels(float(E_J_squid(phi_ext)), E_C_eff, n_max=NQ_MAX)

    preds = {}
    for (nq, na, nr), E_sweet in sweet_energy_by_label.items():
        preds[(nq, na, nr)] = float(E_sweet + (eq_phi[nq] - eq_ref[nq]))
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
    n_curr, n_prev = overlap.shape
    cost = overlap_weight * (1.0 - overlap)

    energy_cost = np.abs(curr_evals[:, None] - prev_evals[None, :]) / energy_scale
    cost += energy_weight * energy_cost

    if prev_prev_evals is not None:
        prev_slope = (prev_evals - prev_prev_evals) / delta_phi
        predicted = prev_evals + prev_slope * delta_phi
        slope_cost = np.abs(curr_evals[:, None] - predicted[None, :]) / slope_scale
        cost += slope_weight * slope_cost

    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.empty_like(col_ind)
    perm[col_ind] = row_ind
    return perm


def track_eigenbranches(raw_evals, all_evecs, phi_vals, start_idx):
    n_flux, n_levels = raw_evals.shape
    tracked_evals = np.zeros_like(raw_evals)
    tracked_evecs = [None] * n_flux

    tracked_evals[start_idx] = raw_evals[start_idx]
    tracked_evecs[start_idx] = all_evecs[start_idx]

    for i in range(start_idx + 1, n_flux):
        prev_prev_evals = tracked_evals[i - 2] if i - 2 >= start_idx else None
        delta_phi = float(phi_vals[i] - phi_vals[i - 1])
        perm = _match_branches_step(
            tracked_evals[i - 1],
            raw_evals[i],
            tracked_evecs[i - 1],
            all_evecs[i],
            delta_phi,
            prev_prev_evals=prev_prev_evals,
        )
        tracked_evals[i] = raw_evals[i][perm]
        tracked_evecs[i] = all_evecs[i][:, perm]

    for i in range(start_idx - 1, -1, -1):
        prev_prev_evals = tracked_evals[i + 2] if i + 2 <= start_idx else None
        delta_phi = float(phi_vals[i] - phi_vals[i + 1])
        perm = _match_branches_step(
            tracked_evals[i + 1],
            raw_evals[i],
            tracked_evecs[i + 1],
            all_evecs[i],
            delta_phi,
            prev_prev_evals=prev_prev_evals,
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
    """
    E = np.asarray(evals_sweet, dtype=float)
    E = E - E[0]
    E_max = float(E[-1])

    if omega_a_hint is not None:
        omega_a_guess = float(omega_a_hint)
    else:
        E_J_sw = float(E_J_squid(phi_sweet))
        eq_sw  = transmon_energy_levels(E_J_sw, E_C_eff, n_max=NQ_MAX)
        omega_a_guess = OMEGA_R_GHZ
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

    x0 = [omega_a_guess, 0.0, 0.0, 0.0, 0.0]

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


# ── Build circuit ──────────────────────────────────────────────────────────────
print("\nBuilding circuit from design_graph.txt...")
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

print("\ndoing tracking..")
all_evals, all_evecs_tracked = track_eigenbranches(raw_evals, all_evecs, phi_vals, mid_idx)

print("Tracking complete.")

# ── Seed labels at sweet spot using identification-style logic ────────────────
print("\nIdentifying sweet-spot levels using fundamental-frequency anchors...")
sweet_params = fit_effective_params_3mode(
    all_evals[mid_idx],
    omega_r_hint=OMEGA_R_GHZ,
    omega_q_hint=OMEGA_Q_GHZ,
    omega_a_hint=OMEGA_A_GHZ,
)
sweet_assignments = assign_labels_3mode(all_evals[mid_idx], sweet_params, include_chi=True)
sweet_labels = {a["k"]: (a["nq"], a["na"], a["nr"]) for a in sweet_assignments}
sweet_energy_by_label = {(a["nq"], a["na"], a["nr"]): a["E"] for a in sweet_assignments}

# ── Fit 3-mode analytical params at sweet spot ────────────────────────────────
print("\nFitting 3-mode analytical parameters at sweet spot...")
analytical_params = fit_analytical_params_3mode(
    all_evals[mid_idx], phi_sweet=phi_vals[mid_idx], omega_a_hint=OMEGA_A_GHZ
)
print(f"  omega_a = {analytical_params['omega_a']:.4f} GHz")
print(f"  alpha_a = {analytical_params['alpha_a'] * 1e3:.1f} MHz")
print(f"  chi_qa  = {analytical_params['chi_qa'] * 1e3:.2f} MHz")
print(f"  chi_ar  = {analytical_params['chi_ar'] * 1e3:.2f} MHz")
print(f"  chi_qr  = {analytical_params['chi_qr'] * 1e3:.2f} MHz")

# ── Attach one label to each tracked branch ───────────────────────────────────
print("Attaching sweet-spot labels to tracked branches")
tracked_branch_labels = {k: sweet_labels.get(k, (-1, -1, -1)) for k in range(N_LEVELS)}

print(f"\nState labels at Phi = {phi_vals[mid_idx]:.3f} Phi_0 (sweet spot):")
for k in range(1, N_LEVELS):
    nq, na, nr = tracked_branch_labels.get(k, (-1, -1, -1))
    print(f"tracked level {k:2d} -> |{nq},{na},{nr}>, E = {all_evals[mid_idx, k]:.4f} GHz")

# ── Plotting ──────────────────────────────────────────────────────────────────
all_qn_labels = set()
all_qn_labels.update(tracked_branch_labels.values())

all_qn_labels.discard((0, 0, 0))
all_qn_labels = sorted(all_qn_labels, key=lambda x: (x[0], x[1], x[2]))

n_colors = len(all_qn_labels)
cmap     = plt.cm.tab20
colors   = {lbl: cmap(i / max(n_colors - 1, 1)) for i, lbl in enumerate(all_qn_labels)}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left panel: numerical spectrum with labels
ax = axes[0]
for lbl in all_qn_labels:
    nq, na, nr = lbl
    E_curve = np.full(N_FLUX, np.nan)
    for i in range(N_FLUX):
        for k in range(1, N_LEVELS):
            if tracked_branch_labels.get(k) == lbl:
                E_curve[i] = all_evals[i, k]
                break
    if not np.all(np.isnan(E_curve)):
        ax.plot(
            phi_vals, E_curve,
            color=colors[lbl], linewidth=1.4,
            label=f"|{nq},{na},{nr}>",
        )

ax.set_xlabel(r"External flux  $\Phi_\mathrm{ext}/\Phi_0$", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel("Energy (GHz)", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_title("Energy Levels vs External Flux", fontsize=TITLE_FONTSIZE)
ax.legend(frameon=False, ncols=2, fontsize=LEGEND_FONTSIZE, loc="upper right")
ax.grid(True, alpha=0.25)

# Right panel: analytical 3-mode model
ax2 = axes[1]
for lbl in all_qn_labels:
    nq, na, nr = lbl
    E_pred = np.array([
        predict_3mode_from_sweet_anchor(
            phi,
            sweet_energy_by_label,
            phi_ref=float(phi_vals[mid_idx]),
        ).get((nq, na, nr), np.nan)
        for phi in phi_vals
    ])
    ax2.plot(
        phi_vals, E_pred,
        color=colors[lbl], linewidth=1.4, linestyle="--",
        label=f"|{nq},{na},{nr}>",
    )

ax2.set_xlabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_ylabel("Energy (GHz)", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_title("Sweet-spot anchored three-mode trend", fontsize=TITLE_FONTSIZE)
ax2.legend(frameon=False, ncols=2, fontsize=LEGEND_FONTSIZE, loc="upper right")
ax2.grid(True, alpha=0.25)

ymax = max(ax.get_ylim()[1], ax2.get_ylim()[1])
for a in axes:
    a.set_ylim(-0.3, ymax)
    a.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

E_J_sweet     = float(E_J_squid(0.0))
omega_q_sweet = float(transmon_energy_levels(E_J_sweet, E_C_eff, n_max=2)[1])
alpha_q_sweet = float(
    2 * transmon_energy_levels(E_J_sweet, E_C_eff, n_max=2)[1]
    - transmon_energy_levels(E_J_sweet, E_C_eff, n_max=2)[2]
)

print("\n--- Analytical three-mode model parameters ---")
print(f"Resonator: omega_r/2pi = {OMEGA_R_GHZ:.3f} GHz")
print(f"Qubit at sweet spot (Phi=0):")
print(f"  E_J^eff = {E_J_sweet:.3f} GHz")
print(f"  E_C     = {E_C_eff:.4f} GHz")
print(f"  omega_q/2pi = {omega_q_sweet:.3f} GHz")
print(f"  alpha_q = {alpha_q_sweet * 1e3:.1f} MHz")
print(f"Ancilla (fitted from spectrum):")
print(f"  omega_a/2pi = {analytical_params['omega_a']:.3f} GHz")
print(f"  alpha_a = {analytical_params['alpha_a'] * 1e3:.1f} MHz")
print(f"Cross-Kerr couplings (fitted):")
print(f"  chi_qr = {analytical_params['chi_qr'] * 1e3:.2f} MHz")
print(f"  chi_qa = {analytical_params['chi_qa'] * 1e3:.2f} MHz")
print(f"  chi_ar = {analytical_params['chi_ar'] * 1e3:.2f} MHz")

fig.suptitle("Flux sweep plots", fontsize=SUPTITLE_FONTSIZE, y=0.93)
fig.tight_layout()

os.makedirs("plot_output", exist_ok=True)
out_path = "plot_output/imet_energy_levels_labelled_v2.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
plt.show()

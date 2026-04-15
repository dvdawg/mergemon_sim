import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scqubits as scq
from scipy.optimize import linear_sum_assignment


AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 14
LEGEND_FONTSIZE = 10
SUPTITLE_FONTSIZE = 18

L_r = 0.264e-9
C_r = 0.878e-12
L_c = 0.396e-9
L_J1 = 30.0e-9
L_J2 = 30.0e-9
C_J1 = 40e-15
C_J2 = 40e-15

PHI0 = 2.067833848e-15
E_CHARGE = 1.602176634e-19
H_PLANCK = 6.62607015e-34

NQ_MAX = 4
NR_MAX = 6
N_LEVELS = 12
N_FLUX = 101
DERIVATIVE_JUMP_WARN_THRESHOLD = 2.0
DERIVATIVE_REPAIR_MIN_IMPROVEMENT = 0.25
DERIVATIVE_REPAIR_MAX_PASSES = 50
DERIVATIVE_REPAIR_WINDOW_PHI = 0.05
FORCE_BEST_DERIVATIVE_REPAIR_SWAP = True
LABEL_VALID_PHI_MAX = 0.45


def inductive_energy_ghz(L):
    return ((PHI0 / (2 * np.pi)) ** 2 / L) / (H_PLANCK * 1e9)


def charging_energy_ghz(C):
    return (E_CHARGE**2 / (2 * C)) / (H_PLANCK * 1e9)


E_J1_ghz = inductive_energy_ghz(L_J1)
E_J2_ghz = inductive_energy_ghz(L_J2)
E_L_c_ghz = inductive_energy_ghz(L_c)
E_L_r_ghz = inductive_energy_ghz(L_r)
E_C1_ghz = charging_energy_ghz(C_J1)
E_C2_ghz = charging_energy_ghz(C_J2)
E_C_r_ghz = charging_energy_ghz(C_r)

L_r_eff = L_r + L_c
OMEGA_R = 1.0 / np.sqrt(L_r_eff * C_r)
OMEGA_R_GHZ = OMEGA_R / (2 * np.pi * 1e9)

C_J_eff = C_J1 + C_J2
E_C_eff = charging_energy_ghz(C_J_eff)
d_asym = (E_J1_ghz - E_J2_ghz) / (E_J1_ghz + E_J2_ghz)


def e_j_squid(phi_ext: float | np.ndarray) -> np.ndarray:
    phi_ext = np.asarray(phi_ext, dtype=float)
    arg = np.pi * phi_ext
    cos_val = np.cos(arg)
    with np.errstate(divide="ignore", invalid="ignore"):
        tan_val = np.tan(arg)
        e_j = (E_J1_ghz + E_J2_ghz) * np.abs(cos_val) * np.sqrt(
            1.0 + d_asym**2 * tan_val**2
        )
    return np.where(np.isfinite(e_j), e_j, 0.0)


def transmon_energy_levels(e_j: float, e_c: float, n_max: int = 5) -> np.ndarray:
    if e_j <= 0:
        n = np.arange(n_max + 1, dtype=float)
        return n * e_c
    n = np.arange(n_max + 1, dtype=float)
    e_n = (
        np.sqrt(8.0 * e_j * e_c) * (n + 0.5)
        - (e_c / 12.0) * (6 * n**2 + 6 * n + 3)
        - e_j
    )
    return e_n - e_n[0]


def predict_2mode(phi_ext: float, chi: float = 0.0) -> dict:
    e_j = float(e_j_squid(phi_ext))
    e_q = transmon_energy_levels(e_j, E_C_eff, n_max=NQ_MAX)
    preds = {}
    for nq in range(NQ_MAX + 1):
        for nr in range(NR_MAX + 1):
            preds[(nq, nr)] = e_q[nq] + nr * OMEGA_R_GHZ + chi * nq * nr
    return preds


def assign_labels(evals_rel: np.ndarray, phi_ext: float, chi: float) -> dict:
    preds = predict_2mode(phi_ext, chi)
    cands = sorted(preds.items(), key=lambda kv: kv[1])
    n_evals = len(evals_rel)
    n_cands = min(len(cands), n_evals)

    cost = np.full((n_evals, n_cands), 1e9)
    for i, e_meas in enumerate(evals_rel):
        for j, ((nq, nr), e_pred) in enumerate(cands[:n_cands]):
            cost[i, j] = abs(e_meas - e_pred)

    row_ind, col_ind = linear_sum_assignment(cost)

    labels = {}
    for row, col in zip(row_ind, col_ind):
        (nq, nr), e_pred = cands[col]
        labels[row] = (nq, nr, e_pred)
    return labels


def chi_from_dressed_double_difference(
    evals_sweet: np.ndarray, phi_sweet: float = 0.0
) -> float:
    """
    Compute chi from dressed energies at sweet spot using:
      chi_10 = E_11 - E_10 - E_01 + E_00
    Energies are in GHz, so returned chi is also in GHz.
    """
    energies = np.asarray(evals_sweet, dtype=float)
    energies = energies - energies[0]

    # Use chi=0 labels only to identify the dressed branches |1,0>, |0,1>, |1,1>.
    # The extracted chi itself comes only from dressed energies.
    label_info = assign_labels(energies, phi_sweet, chi=0.0)
    label_to_level = {(nq, nr): level_idx for level_idx, (nq, nr, _) in label_info.items()}

    required_labels = [(0, 0), (1, 0), (0, 1), (1, 1)]
    missing = [lbl for lbl in required_labels if lbl not in label_to_level]
    if missing:
        missing_str = ", ".join(f"|{nq},{nr}>" for nq, nr in missing)
        raise RuntimeError(
            f"Could not identify required dressed states at sweet spot: {missing_str}"
        )

    idx_00 = label_to_level[(0, 0)]
    idx_10 = label_to_level[(1, 0)]
    idx_01 = label_to_level[(0, 1)]
    idx_11 = label_to_level[(1, 1)]

    e00 = energies[idx_00]
    e10 = energies[idx_10]
    e01 = energies[idx_01]
    e11 = energies[idx_11]

    return float(e11 - e10 - e01 + e00)


def _match_branches_step(
    prev_evals,
    curr_evals,
    prev_evecs,
    curr_evecs,
    delta_phi,
    prev_prev_evals=None,
    next_evals=None,
    next_delta_phi=None,
    overlap_weight=4.0,
    energy_weight=0.7,
    derivative_weight=2.5,
    derivative_lookahead_weight=2.5,
    energy_scale=0.05,
    derivative_scale=0.5,
    derivative_hard_threshold=DERIVATIVE_JUMP_WARN_THRESHOLD,
    derivative_hard_penalty=1.0e6,
):
    overlap = np.abs(curr_evecs.conj().T @ prev_evecs) ** 2
    cost = overlap_weight * (1.0 - overlap)
    energy_cost = np.abs(curr_evals[:, None] - prev_evals[None, :]) / energy_scale
    cost += energy_weight * energy_cost

    if prev_prev_evals is not None:
        prev_slope = (prev_evals - prev_prev_evals) / delta_phi
        curr_slope = (curr_evals[:, None] - prev_evals[None, :]) / delta_phi
        abs_derivative_jump = np.abs(curr_slope - prev_slope[None, :])
        derivative_jump_cost = (abs_derivative_jump / derivative_scale) ** 2
        cost += derivative_weight * derivative_jump_cost
        cost += np.where(
            abs_derivative_jump >= derivative_hard_threshold,
            derivative_hard_penalty
            * (abs_derivative_jump / derivative_hard_threshold) ** 2,
            0.0,
        )

    if next_evals is not None and next_delta_phi is not None:
        curr_slope = (curr_evals[:, None] - prev_evals[None, :]) / delta_phi
        next_slope = (
            next_evals[:, None, None] - curr_evals[None, :, None]
        ) / next_delta_phi
        best_next_jump = np.min(np.abs(next_slope - curr_slope[None, :, :]), axis=0)
        lookahead_cost = (best_next_jump / derivative_scale) ** 2
        cost += derivative_lookahead_weight * lookahead_cost
        cost += np.where(
            best_next_jump >= derivative_hard_threshold,
            derivative_hard_penalty
            * (best_next_jump / derivative_hard_threshold) ** 2,
            0.0,
        )

    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.empty_like(col_ind)
    perm[col_ind] = row_ind
    return perm


def _derivative_jump_metrics(
    evals,
    phi_vals,
    valid_phi_max=LABEL_VALID_PHI_MAX,
    threshold=DERIVATIVE_JUMP_WARN_THRESHOLD,
):
    slopes = np.diff(evals, axis=0) / np.diff(phi_vals)[:, None]
    jumps = np.abs(np.diff(slopes, axis=0))
    jump_centers = phi_vals[1:-1]
    valid = np.abs(jump_centers) <= valid_phi_max
    valid_jumps = jumps[valid]
    if valid_jumps.size == 0:
        return {"score": 0.0, "count": 0, "max": 0.0}

    excess = np.maximum(valid_jumps - threshold, 0.0)
    return {
        "score": float(np.sum(excess**2)),
        "count": int(np.count_nonzero(excess > 0.0)),
        "max": float(np.max(valid_jumps)),
    }


def _local_derivative_jump_metrics(
    evals,
    phi_vals,
    branches,
    centers,
    window_phi=DERIVATIVE_REPAIR_WINDOW_PHI,
    threshold=DERIVATIVE_JUMP_WARN_THRESHOLD,
):
    slopes = np.diff(evals[:, branches], axis=0) / np.diff(phi_vals)[:, None]
    jumps = np.abs(np.diff(slopes, axis=0))
    jump_centers = phi_vals[1:-1]
    local = np.zeros_like(jump_centers, dtype=bool)
    for center in centers:
        local |= np.abs(jump_centers - center) <= window_phi

    local_jumps = jumps[local]
    if local_jumps.size == 0:
        return {"score": 0.0, "count": 0, "max": 0.0}

    excess = np.maximum(local_jumps - threshold, 0.0)
    return {
        "score": float(np.sum(excess**2)),
        "count": int(np.count_nonzero(excess > 0.0)),
        "max": float(np.max(local_jumps)),
    }


def _symmetric_interval_for_phi(phi_vals, phi_abs):
    lo = int(np.argmin(np.abs(phi_vals + phi_abs)))
    hi = int(np.argmin(np.abs(phi_vals - phi_abs)))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _swap_branch_interval(evals, lo, hi, a, b):
    swapped = np.array(evals, copy=True)
    a_values = np.array(swapped[lo : hi + 1, a], copy=True)
    b_values = np.array(swapped[lo : hi + 1, b], copy=True)
    swapped[lo : hi + 1, a] = b_values
    swapped[lo : hi + 1, b] = a_values
    return swapped


def repair_derivative_swaps(
    tracked_evals,
    tracked_evecs,
    phi_vals,
    start_idx,
    valid_phi_max=LABEL_VALID_PHI_MAX,
    min_improvement=DERIVATIVE_REPAIR_MIN_IMPROVEMENT,
    max_passes=DERIVATIVE_REPAIR_MAX_PASSES,
):
    repaired_evals = np.array(tracked_evals, copy=True)
    repaired_evecs = [np.array(evec, copy=True) for evec in tracked_evecs]
    n_flux, n_levels = repaired_evals.shape
    slopes = np.diff(repaired_evals, axis=0) / np.diff(phi_vals)[:, None]
    jumps = np.abs(np.diff(slopes, axis=0))
    jump_candidates = np.argwhere(jumps >= DERIVATIVE_JUMP_WARN_THRESHOLD)
    jump_candidates = sorted(
        jump_candidates,
        key=lambda item: jumps[item[0], item[1]],
        reverse=True,
    )

    chosen = None
    for jump_idx, a in jump_candidates:
        jump_center_idx = int(jump_idx + 1)
        phi = float(phi_vals[jump_center_idx])
        if not (-valid_phi_max <= phi < 0.0):
            continue

        energies_at_jump = repaired_evals[jump_center_idx]
        gaps = np.abs(energies_at_jump - energies_at_jump[int(a)])
        gaps[0] = np.inf
        gaps[int(a)] = np.inf
        b = int(np.argmin(gaps))
        phi_abs = abs(phi)
        lo, hi = _symmetric_interval_for_phi(phi_vals, phi_abs)
        if lo == 0 or hi == n_flux - 1:
            continue

        chosen = {
            "jump_idx": int(jump_idx),
            "jump_center_idx": jump_center_idx,
            "jump": float(jumps[jump_idx, a]),
            "a": int(a),
            "b": b,
            "lo": lo,
            "hi": hi,
            "gap": float(gaps[b]),
        }
        break

    if chosen is None:
        print(
            "Derivative swap repair: no negative-flux derivative jumps above "
            f"{DERIVATIVE_JUMP_WARN_THRESHOLD:.3g} GHz/Phi0 found inside "
            f"|Phi| <= {valid_phi_max:.3g}."
        )
        return repaired_evals, repaired_evecs

    a = chosen["a"]
    b = chosen["b"]
    lo = chosen["lo"]
    hi = chosen["hi"]
    probe_indices = sorted({lo, chosen["jump_center_idx"], start_idx, hi})
    before_probe = [
        (float(phi_vals[idx]), float(repaired_evals[idx, a]), float(repaired_evals[idx, b]))
        for idx in probe_indices
    ]

    repaired_evals = _swap_branch_interval(repaired_evals, lo, hi, a, b)
    for flux_idx in range(lo, hi + 1):
        a_vec = np.array(repaired_evecs[flux_idx][:, a], copy=True)
        b_vec = np.array(repaired_evecs[flux_idx][:, b], copy=True)
        repaired_evecs[flux_idx][:, a] = b_vec
        repaired_evecs[flux_idx][:, b] = a_vec

    after_probe = [
        (float(phi_vals[idx]), float(repaired_evals[idx, a]), float(repaired_evals[idx, b]))
        for idx in probe_indices
    ]

    print("Derivative swap repair: force-applied closest-energy symmetric swap.")
    print(
        "  "
        f"jump phi={phi_vals[chosen['jump_center_idx']]:+.4f}, "
        f"|Delta slope|={chosen['jump']:.3g} GHz/Phi0, "
        f"branches {a}<->{b}, closest gap={chosen['gap']:.3g} GHz"
    )
    print(
        "  "
        f"swapped interval [{phi_vals[lo]:+.4f}, {phi_vals[hi]:+.4f}] "
        f"({hi - lo + 1} flux points)"
    )
    print("  probe before swap:")
    for phi, ea, eb in before_probe:
        print(f"    phi={phi:+.4f}: branch {a}={ea:.6g}, branch {b}={eb:.6g}")
    print("  probe after swap:")
    for phi, ea, eb in after_probe:
        print(f"    phi={phi:+.4f}: branch {a}={ea:.6g}, branch {b}={eb:.6g}")

    return repaired_evals, repaired_evecs


def track_eigenbranches(raw_evals, all_evecs, phi_vals, start_idx):
    n_flux, _ = raw_evals.shape
    tracked_evals = np.zeros_like(raw_evals)
    tracked_evecs = [None] * n_flux

    tracked_evals[start_idx] = raw_evals[start_idx]
    tracked_evecs[start_idx] = all_evecs[start_idx]

    for i in range(start_idx + 1, n_flux):
        prev_prev_evals = tracked_evals[i - 2] if i - 2 >= start_idx else None
        delta_phi = float(phi_vals[i] - phi_vals[i - 1])
        next_evals = raw_evals[i + 1] if i + 1 < n_flux else None
        next_delta_phi = (
            float(phi_vals[i + 1] - phi_vals[i]) if i + 1 < n_flux else None
        )
        perm = _match_branches_step(
            tracked_evals[i - 1],
            raw_evals[i],
            tracked_evecs[i - 1],
            all_evecs[i],
            delta_phi,
            prev_prev_evals=prev_prev_evals,
            next_evals=next_evals,
            next_delta_phi=next_delta_phi,
        )
        tracked_evals[i] = raw_evals[i][perm]
        tracked_evecs[i] = all_evecs[i][:, perm]

    for i in range(start_idx - 1, -1, -1):
        prev_prev_evals = tracked_evals[i + 2] if i + 2 <= start_idx else None
        delta_phi = float(phi_vals[i] - phi_vals[i + 1])
        next_evals = raw_evals[i - 1] if i - 1 >= 0 else None
        next_delta_phi = (
            float(phi_vals[i - 1] - phi_vals[i]) if i - 1 >= 0 else None
        )
        perm = _match_branches_step(
            tracked_evals[i + 1],
            raw_evals[i],
            tracked_evecs[i + 1],
            all_evecs[i],
            delta_phi,
            prev_prev_evals=prev_prev_evals,
            next_evals=next_evals,
            next_delta_phi=next_delta_phi,
        )
        tracked_evals[i] = raw_evals[i][perm]
        tracked_evecs[i] = all_evecs[i][:, perm]

    return repair_derivative_swaps(tracked_evals, tracked_evecs, phi_vals, start_idx)


def report_derivative_jumps(
    tracked_evals,
    phi_vals,
    threshold=DERIVATIVE_JUMP_WARN_THRESHOLD,
    max_reports=12,
):
    slopes = np.diff(tracked_evals, axis=0) / np.diff(phi_vals)[:, None]
    slope_jumps = np.diff(slopes, axis=0)
    abs_jumps = np.abs(slope_jumps)
    if abs_jumps.size == 0:
        return []

    candidates = np.argwhere(abs_jumps >= threshold)
    records = []
    for flux_minus_one, level_idx in candidates:
        records.append(
            {
                "level": int(level_idx),
                "flux_index": int(flux_minus_one + 1),
                "phi": float(phi_vals[flux_minus_one + 1]),
                "jump": float(abs_jumps[flux_minus_one, level_idx]),
                "slope_before": float(slopes[flux_minus_one, level_idx]),
                "slope_after": float(slopes[flux_minus_one + 1, level_idx]),
            }
        )

    records.sort(key=lambda rec: rec["jump"], reverse=True)
    if not records:
        print(
            f"Derivative smoothness check: no branch slope jumps above "
            f"{threshold:.3g} GHz/Phi0."
        )
        return []

    print(
        f"Derivative smoothness check: {len(records)} branch slope jumps above "
        f"{threshold:.3g} GHz/Phi0."
    )
    for rec in records[:max_reports]:
        print(
            "  "
            f"level={rec['level']:2d}, "
            f"phi={rec['phi']:+.4f}, "
            f"|Delta slope|={rec['jump']:.3g} GHz/Phi0, "
            f"before={rec['slope_before']:.3g}, "
            f"after={rec['slope_after']:.3g}"
        )
    if len(records) > max_reports:
        print(f"  ... {len(records) - max_reports} more not shown")
    return records


def normalize_evecs(evec):
    evec = np.array(evec)
    if evec.ndim == 1:
        evec = evec[:, np.newaxis]
    if evec.shape[0] < evec.shape[1]:
        evec = evec.T
    return evec


def build_circuit():
    imet_yaml = f"""# iMET: asymmetric SQUID transmon coupled to LC resonator
branches:
- ["JJ", 1,4, {E_J1_ghz:.6g}, {E_C1_ghz:.6g}]
- ["JJ", 1,2, {E_J2_ghz:.6g}, {E_C2_ghz:.6g}]
- ["L",  2,4, {E_L_c_ghz:.6g}]
- ["L",  2,3, {E_L_r_ghz:.6g}]
- ["C",  3,4, {E_C_r_ghz:.6g}]
"""

    circ = scq.Circuit(imet_yaml, from_file=False, ext_basis="harmonic")
    circ.cutoff_n_1 = 6
    circ.cutoff_ext_2 = 10
    circ.cutoff_ext_3 = 10
    return circ


def choose_squid_flux_attr(circ):
    flux_syms = circ.external_fluxes
    flux_attrs = [str(sym) for sym in flux_syms]
    if not flux_attrs:
        sys.exit("No external flux variables found - check circuit topology.")

    print(f"External flux variables: {flux_attrs}")
    for attr in flux_attrs:
        setattr(circ, attr, 0.0)

    ev_base, _ = circ.eigensys(evals_count=5)
    e1_base = (ev_base - ev_base[0])[1]
    squid_attr = flux_attrs[0]

    if len(flux_attrs) > 1:
        best_shift = -1.0
        for attr in flux_attrs:
            for reset_attr in flux_attrs:
                setattr(circ, reset_attr, 0.0)
            setattr(circ, attr, 0.5)
            ev_test, _ = circ.eigensys(evals_count=5)
            shift = abs((ev_test - ev_test[0])[1] - e1_base)
            print(f"{attr} = 0.5 Phi0 -> |Delta f_01| = {shift:.4f} GHz")
            if shift > best_shift:
                best_shift = shift
                squid_attr = attr
        for attr in flux_attrs:
            setattr(circ, attr, 0.0)

    print(f"SQUID flux variable: '{squid_attr}'")
    for attr in flux_attrs:
        if attr != squid_attr:
            setattr(circ, attr, 0.0)
    return squid_attr


def collect_flux_sweep(circ, squid_attr, phi_vals):
    raw_evals = np.zeros((len(phi_vals), N_LEVELS))
    all_evecs = []

    print(f"\nSweeping {len(phi_vals)} flux points, {N_LEVELS} eigenvalues each...")
    for i, phi in enumerate(phi_vals):
        setattr(circ, squid_attr, float(phi))
        evals, evecs = circ.eigensys(evals_count=N_LEVELS)
        raw_evals[i] = evals - evals[0]
        all_evecs.append(normalize_evecs(evecs))
        if (i + 1) % 20 == 0 or i == 0 or i == len(phi_vals) - 1:
            print(f"  {i + 1:3d} / {len(phi_vals)}")

    return raw_evals, all_evecs


def label_flux_points(tracked_evals, phi_vals, chi_fit, valid_phi_max=LABEL_VALID_PHI_MAX):
    level_labels_flux = []
    for i, phi in enumerate(phi_vals):
        if abs(phi) > valid_phi_max:
            level_labels_flux.append({})
            continue
        e_rel = tracked_evals[i] - tracked_evals[i, 0]
        label_info = assign_labels(e_rel, phi, chi_fit)
        level_labels_flux.append(
            {k: (nq, nr) for k, (nq, nr, _) in label_info.items()}
        )
    return level_labels_flux


def sweet_spot_branch_labels(level_labels_flux, mid_idx):
    labels = {}
    sweet_labels = level_labels_flux[mid_idx]
    for level_idx in range(1, N_LEVELS):
        labels[level_idx] = sweet_labels.get(level_idx)
    return labels


def plot_labelled_sweep(phi_vals, tracked_evals, branch_labels, chi_fit):
    plotted_branches = list(range(1, N_LEVELS))
    n_colors = len(plotted_branches)
    cmap = plt.cm.tab20
    colors = {
        level_idx: cmap(i / max(n_colors - 1, 1))
        for i, level_idx in enumerate(plotted_branches)
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for level_idx in plotted_branches:
        label = branch_labels.get(level_idx)
        legend_label = (
            f"branch {level_idx}: unlabeled"
            if label is None
            else f"branch {level_idx}: |{label[0]},{label[1]}>"
        )
        ax.plot(
            phi_vals,
            tracked_evals[:, level_idx],
            color=colors[level_idx],
            linewidth=1.4,
            label=legend_label,
        )

    ax.set_xlabel(
        r"External flux  $\Phi_\mathrm{ext}/\Phi_0$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.set_ylabel("Energy (GHz)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Energy Levels vs External Flux", fontsize=TITLE_FONTSIZE)
    ax.legend(frameon=False, ncols=2, fontsize=LEGEND_FONTSIZE, loc="upper right")
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    plotted_labels = sorted(
        {label for label in branch_labels.values() if label is not None and label != (0, 0)},
        key=lambda label: (label[0], label[1]),
    )
    for i, label in enumerate(plotted_labels):
        nq, nr = label
        e_pred = np.array(
            [predict_2mode(phi, chi=chi_fit).get((nq, nr), np.nan) for phi in phi_vals]
        )
        color = cmap(i / max(len(plotted_labels) - 1, 1))
        ax2.plot(
            phi_vals,
            e_pred,
            color=color,
            linewidth=1.4,
            linestyle="--",
            label=f"|{nq},{nr}>",
        )

    ax2.set_xlabel(
        r"External flux $\Phi_\mathrm{ext}/\Phi_0$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax2.set_ylabel("Energy (GHz)", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_title("Analytical two-mode model", fontsize=TITLE_FONTSIZE)
    ax2.legend(frameon=False, ncols=2, fontsize=LEGEND_FONTSIZE, loc="upper right")
    ax2.grid(True, alpha=0.25)

    ymax = max(ax.get_ylim()[1], ax2.get_ylim()[1])
    for axis in axes:
        axis.set_ylim(-0.3, ymax)
        axis.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    fig.suptitle("Flux sweep plots", fontsize=SUPTITLE_FONTSIZE, y=0.93)
    fig.tight_layout()
    return fig


def print_model_summary(chi_fit):
    e_j_sweet = float(e_j_squid(0.0))
    sweet_levels = transmon_energy_levels(e_j_sweet, E_C_eff, n_max=2)
    omega_q_sweet = float(sweet_levels[1])
    alpha_sweet = float(sweet_levels[2] - 2 * sweet_levels[1])

    print("\n--- Analytical two-mode model parameters ---")
    print(f"Resonator: omega_r/2pi = {OMEGA_R_GHZ:.3f} GHz")
    print(f"  L_eff = L_r + L_c = {L_r_eff * 1e9:.2f} nH, C_r = {C_r * 1e12:.1f} pF")
    print("Qubit at sweet spot (Phi=0):")
    print(f"  E_J_eff = {e_j_sweet:.3f} GHz")
    print(f"  E_C = {E_C_eff:.4f} GHz")
    print(f"  omega_q/2pi = {omega_q_sweet:.3f} GHz")
    print(f"  alpha_q = {alpha_sweet * 1e3:.1f} MHz")
    print(f"  chi = {chi_fit * 1e3:.2f} MHz")

    print(f"\nResonator (SHO): omega_r/2pi = {OMEGA_R_GHZ:.4f} GHz")
    print(f"L_eff = L_r + L_c = {L_r_eff * 1e9:.3f} nH")
    print(f"C_r = {C_r * 1e12:.3f} pF")
    print("\nQubit (SQUID transmon):")
    print(f"C_eff = C_J1+C_J2 = {C_J_eff * 1e15:.1f} fF")
    print(f"E_C = {E_C_eff:.4f} GHz")
    print(f"d (asymmetry) = {d_asym:.4f}")
    print(f"E_J(Phi=0) = {e_j_sweet:.4f} GHz")
    print(f"omega_q(Phi=0)/2pi = {omega_q_sweet:.4f} GHz")
    print(f"alpha_q(Phi=0) = {alpha_sweet * 1e3:.1f} MHz")
    print("\nDispersive coupling (fit):")
    print(f"chi/2pi = {chi_fit * 1e3:.2f} MHz")

def main():
    print(f"Resonator mode: L_eff = {L_r_eff * 1e9:.3f} nH, C_r = {C_r * 1e12:.2f} pF")
    print(f"omega_r / 2pi = {OMEGA_R_GHZ:.4f} GHz")
    print(f"\nQubit mode: C_eff = {C_J_eff * 1e15:.1f} fF")
    print(f"E_C = {E_C_eff:.4f} GHz")
    print(f"E_J1 = {E_J1_ghz:.4f} GHz, E_J2 = {E_J2_ghz:.4f} GHz")
    print(f"asymmetry d = {d_asym:.4f}")

    print("\nBuilding circuit...")
    circ = build_circuit()
    squid_attr = choose_squid_flux_attr(circ)

    phi_vals = np.linspace(-0.5, 0.5, N_FLUX)
    mid_idx = N_FLUX // 2

    raw_evals, all_evecs = collect_flux_sweep(circ, squid_attr, phi_vals)

    print("\nTracking eigenbranches with heatmap sweep rule...")
    tracked_evals, _ = track_eigenbranches(raw_evals, all_evecs, phi_vals, mid_idx)
    print("Tracking complete.")
    report_derivative_jumps(tracked_evals, phi_vals)

    print("\nComputing dispersive coupling chi from dressed-energy double difference")
    chi_fit = chi_from_dressed_double_difference(
        tracked_evals[mid_idx], phi_sweet=phi_vals[mid_idx]
    )
    print(f"chi / 2pi = {chi_fit * 1e3:.2f} MHz")

    print("Assigning |nq, nr> labels at flux points")
    level_labels_flux = label_flux_points(tracked_evals, phi_vals, chi_fit)
    branch_labels = sweet_spot_branch_labels(level_labels_flux, mid_idx)
    print(
        f"Labels assigned only for |Phi| <= {LABEL_VALID_PHI_MAX:.2f} Phi0; "
        "edge labels are treated as unreliable."
    )

    print(f"\nState labels at Phi = {phi_vals[mid_idx]:.3f} Phi0 (sweet spot):")
    for level_idx in range(1, N_LEVELS):
        nq, nr = branch_labels.get(level_idx) or (-1, -1)
        print(
            f"tracked level {level_idx:2d} -> |{nq},{nr}>, "
            f"E = {tracked_evals[mid_idx, level_idx]:.4f} GHz"
        )

    fig = plot_labelled_sweep(phi_vals, tracked_evals, branch_labels, chi_fit)
    print_model_summary(chi_fit)

    os.makedirs("plot_output", exist_ok=True)
    out_path = "plot_output/imet_energy_levels_heatmap_tracked_labelled.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

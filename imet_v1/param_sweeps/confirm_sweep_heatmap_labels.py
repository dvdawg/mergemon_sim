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
DERIVATIVE_JUMP_WARN_THRESHOLD = 5.0
LABEL_VALID_PHI_MAX = 0.45
DERIVATIVE_CHECK_PHI_MAX = 0.45
MAX_DERIVATIVE_SMOOTHING_SWAPS_PER_LEVEL = 4 * N_LEVELS
DERIVATIVE_POST_PASS_DIAGNOSTIC_PARTNERS = 4
DERIVATIVE_POST_PASS_INTERVAL_PAD_MAX = 3
DERIVATIVE_CLOSEST_ENERGY_SEARCH_RADIUS = 3
DERIVATIVE_MAX_CLOSEST_SWAP_GAP_GHZ = 0.75
DERIVATIVE_SMOOTHNESS_NEIGHBORHOOD = 3
DERIVATIVE_REPAIR_MIN_TARGET_IMPROVEMENT = 1.0


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
    overlap_weight=4.0,
    energy_weight=0.7,
    derivative_weight=1.6,
    energy_scale=0.05,
    derivative_scale=2.0,
):
    overlap = np.abs(curr_evecs.conj().T @ prev_evecs) ** 2
    cost = overlap_weight * (1.0 - overlap)
    energy_cost = np.abs(curr_evals[:, None] - prev_evals[None, :]) / energy_scale
    cost += energy_weight * energy_cost

    if prev_prev_evals is not None:
        prev_slope = (prev_evals - prev_prev_evals) / delta_phi
        curr_slope = (curr_evals[:, None] - prev_evals[None, :]) / delta_phi
        derivative_jump_cost = np.abs(curr_slope - prev_slope[None, :]) / derivative_scale
        cost += derivative_weight * derivative_jump_cost

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


def report_derivative_jumps(
    tracked_evals,
    phi_vals,
    threshold=DERIVATIVE_JUMP_WARN_THRESHOLD,
    phi_max=DERIVATIVE_CHECK_PHI_MAX,
    max_reports=12,
):
    records = _neighborhood_derivative_jump_records(
        tracked_evals,
        phi_vals,
        threshold=threshold,
        positive_half_only=False,
        phi_max=phi_max,
    )

    records.sort(key=lambda rec: rec["jump"], reverse=True)
    if not records:
        print(
            f"Derivative smoothness check: no fitted-slope jumps above "
            f"{threshold:.3g} GHz/Phi0 inside |Phi| <= {phi_max:.3g}."
        )
        return []

    print(
        f"Derivative smoothness check: {len(records)} fitted-slope jumps above "
        f"{threshold:.3g} GHz/Phi0 inside |Phi| <= {phi_max:.3g} "
        f"using {DERIVATIVE_SMOOTHNESS_NEIGHBORHOOD}-step neighborhoods."
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


def report_derivative_jump_scale(
    tracked_evals,
    phi_vals,
    positive_half_only=True,
    phi_max=DERIVATIVE_CHECK_PHI_MAX,
    max_reports=24,
):
    records = _neighborhood_derivative_jump_records(
        tracked_evals,
        phi_vals,
        positive_half_only=positive_half_only,
        phi_max=phi_max,
    )
    if not records:
        print("Derivative jump scale: not enough flux points to compute jumps.")
        return []

    half_label = "positive-half" if positive_half_only else "full-sweep"
    selected = np.array([rec["jump"] for rec in records if rec["level"] > 0])
    percentiles = [50, 75, 90, 95, 98, 99, 99.5, 100]
    values = np.percentile(selected, percentiles)
    print(
        f"\nDerivative jump scale ({half_label}, "
        f"|Phi| <= {phi_max:.3g}, "
        f"{DERIVATIVE_SMOOTHNESS_NEIGHBORHOOD}-step fitted slopes, "
        "excluding ground branch):"
    )
    for pct, value in zip(percentiles, values):
        print(f"  p{pct:>4g}: {value:.3g} GHz/Phi0")

    records.sort(key=lambda rec: rec["jump"], reverse=True)

    print(f"  top {min(max_reports, len(records))} jumps:")
    for rec in records[:max_reports]:
        print(
            "    "
            f"level={rec['level']:2d}, phi={rec['phi']:+.4f}, "
            f"|Delta slope|={rec['jump']:.3g}, "
            f"before={rec['slope_before']:.3g}, after={rec['slope_after']:.3g}"
        )
    return records


def _positive_half_derivative_jumps(
    tracked_evals,
    phi_vals,
    level_idx,
    threshold=DERIVATIVE_JUMP_WARN_THRESHOLD,
    phi_max=DERIVATIVE_CHECK_PHI_MAX,
):
    records = _neighborhood_derivative_jump_records(
        tracked_evals,
        phi_vals,
        level_idx=level_idx,
        threshold=threshold,
        positive_half_only=True,
        phi_max=phi_max,
    )
    records.sort(key=lambda rec: rec["jump"], reverse=True)
    return records


def _derivative_jump_record_at(tracked_evals, phi_vals, flux_index, level_idx):
    return _neighborhood_derivative_jump_record_at(
        tracked_evals,
        phi_vals,
        flux_index,
        level_idx,
    )


def _fit_line_slope(x_vals, y_vals):
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    x_centered = x - np.mean(x)
    denom = float(np.dot(x_centered, x_centered))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x_centered, y - np.mean(y)) / denom)


def _neighborhood_derivative_jump_record_at(
    tracked_evals,
    phi_vals,
    flux_index,
    level_idx,
    neighborhood=DERIVATIVE_SMOOTHNESS_NEIGHBORHOOD,
):
    left = int(flux_index) - neighborhood
    right = int(flux_index) + neighborhood
    if left < 0 or right >= len(phi_vals):
        return None

    slope_before = _fit_line_slope(
        phi_vals[left : flux_index + 1],
        tracked_evals[left : flux_index + 1, level_idx],
    )
    slope_after = _fit_line_slope(
        phi_vals[flux_index : right + 1],
        tracked_evals[flux_index : right + 1, level_idx],
    )
    jump = float(abs(slope_after - slope_before))
    return {
        "level": int(level_idx),
        "flux_index": int(flux_index),
        "phi": float(phi_vals[flux_index]),
        "jump": jump,
        "slope_before": float(slope_before),
        "slope_after": float(slope_after),
    }


def _neighborhood_derivative_jump_records(
    tracked_evals,
    phi_vals,
    level_idx=None,
    threshold=None,
    positive_half_only=False,
    phi_max=DERIVATIVE_CHECK_PHI_MAX,
):
    levels = (
        [int(level_idx)]
        if level_idx is not None
        else list(range(tracked_evals.shape[1]))
    )
    records = []
    for flux_idx in range(
        DERIVATIVE_SMOOTHNESS_NEIGHBORHOOD,
        len(phi_vals) - DERIVATIVE_SMOOTHNESS_NEIGHBORHOOD,
    ):
        phi = float(phi_vals[flux_idx])
        if positive_half_only:
            if phi <= 0.0 or phi > phi_max:
                continue
        elif abs(phi) > phi_max:
            continue
        for level in levels:
            rec = _neighborhood_derivative_jump_record_at(
                tracked_evals,
                phi_vals,
                flux_idx,
                level,
            )
            if rec is None:
                continue
            if threshold is not None and rec["jump"] < threshold:
                continue
            records.append(rec)
    return records


def _format_jump_record(record):
    if record is None:
        return "unavailable"
    return (
        f"{record['jump']:.3g} "
        f"(before={record['slope_before']:.3g}, after={record['slope_after']:.3g})"
    )


def _swap_key(level_idx, partner_idx, left_idx, right_idx):
    branch_pair = tuple(sorted((int(level_idx), int(partner_idx))))
    return branch_pair + (int(left_idx), int(right_idx))


def _closest_branch_at_flux(
    tracked_evals,
    flux_idx,
    level_idx,
    left_idx,
    right_idx,
    blocked_swap_keys,
):
    n_levels = tracked_evals.shape[1]
    branch_energy = tracked_evals[flux_idx, level_idx]
    partner_indices = np.array(
        [idx for idx in range(1, n_levels) if idx != level_idx],
        dtype=int,
    )
    if partner_indices.size == 0:
        return None

    energy_diffs = np.abs(tracked_evals[flux_idx, partner_indices] - branch_energy)
    for order_idx in np.argsort(energy_diffs):
        partner_idx = int(partner_indices[order_idx])
        key = _swap_key(level_idx, partner_idx, left_idx, right_idx)
        if key not in blocked_swap_keys:
            return partner_idx
    return None


def _partner_order_for_jump(tracked_evals, flux_idx, level_idx):
    n_levels = tracked_evals.shape[1]
    branch_energy = tracked_evals[flux_idx, level_idx]
    partner_indices = [idx for idx in range(1, n_levels) if idx != level_idx]
    partner_indices.sort(key=lambda idx: abs(tracked_evals[flux_idx, idx] - branch_energy))
    return partner_indices


def _closest_energy_index_near_jump(
    tracked_evals,
    phi_vals,
    level_idx,
    partner_idx,
    flux_idx,
    search_radius=DERIVATIVE_CLOSEST_ENERGY_SEARCH_RADIUS,
    phi_max=DERIVATIVE_CHECK_PHI_MAX,
):
    lo = max(0, int(flux_idx) - search_radius)
    hi = min(len(phi_vals) - 1, int(flux_idx) + search_radius)
    candidate_indices = [
        idx for idx in range(lo, hi + 1) if 0.0 <= phi_vals[idx] <= phi_max
    ]
    if not candidate_indices:
        return int(flux_idx)

    gaps = np.abs(
        tracked_evals[candidate_indices, level_idx]
        - tracked_evals[candidate_indices, partner_idx]
    )
    return int(candidate_indices[int(np.argmin(gaps))])


def _partner_energy_gap_diagnostics(
    tracked_evals,
    phi_vals,
    level_idx,
    flux_idx,
    max_partners=DERIVATIVE_POST_PASS_DIAGNOSTIC_PARTNERS,
):
    records = []
    for partner_idx in range(1, tracked_evals.shape[1]):
        if partner_idx == level_idx:
            continue
        closest_idx = _closest_energy_index_near_jump(
            tracked_evals,
            phi_vals,
            level_idx,
            partner_idx,
            flux_idx,
        )
        records.append(
            (
                abs(
                    tracked_evals[closest_idx, partner_idx]
                    - tracked_evals[closest_idx, level_idx]
                ),
                partner_idx,
                closest_idx,
            )
        )
    records.sort(key=lambda item: item[0])
    return "; ".join(
        f"level {partner_idx} min_gap={gap:.4g} at {phi_vals[closest_idx]:+.4f}"
        for gap, partner_idx, closest_idx in records[:max_partners]
    )


def _swap_columns_for_rows(evals, row_indices, level_idx, partner_idx):
    swapped = np.array(evals, copy=True)
    level_values = swapped[row_indices, level_idx].copy()
    swapped[row_indices, level_idx] = swapped[row_indices, partner_idx]
    swapped[row_indices, partner_idx] = level_values
    return swapped


def _trial_swap_recheck(
    evals,
    phi_vals,
    row_indices,
    level_idx,
    partner_idx,
    flux_index,
    threshold,
):
    trial_evals = _swap_columns_for_rows(evals, row_indices, level_idx, partner_idx)
    level_recheck = _derivative_jump_record_at(
        trial_evals,
        phi_vals,
        flux_index,
        level_idx,
    )
    partner_recheck = _derivative_jump_record_at(
        trial_evals,
        phi_vals,
        flux_index,
        partner_idx,
    )
    passes = (
        level_recheck is not None
        and partner_recheck is not None
        and level_recheck["jump"] < threshold
        and partner_recheck["jump"] < threshold
    )
    return trial_evals, level_recheck, partner_recheck, passes


def _candidate_swap_intervals(n_flux, left_idx, right_idx):
    candidates = []
    seen = set()

    def add(label, lo, hi):
        lo = max(0, int(lo))
        hi = min(n_flux - 1, int(hi))
        if lo > hi or (lo, hi) in seen:
            return
        seen.add((lo, hi))
        candidates.append((label, lo, hi, np.arange(lo, hi + 1)))

    add("base", left_idx, right_idx)
    for pad in range(1, DERIVATIVE_POST_PASS_INTERVAL_PAD_MAX + 1):
        add(f"symmetric-pad-{pad}", left_idx - pad, right_idx + pad)
        add(f"right-pad-{pad}", left_idx, right_idx + pad)
        add(f"left-pad-{pad}", left_idx - pad, right_idx)

    return candidates


def _partner_diagnostics(
    tracked_evals,
    flux_idx,
    level_idx,
    left_idx,
    right_idx,
    blocked_swap_keys,
    max_partners=DERIVATIVE_POST_PASS_DIAGNOSTIC_PARTNERS,
):
    n_levels = tracked_evals.shape[1]
    branch_energy = tracked_evals[flux_idx, level_idx]
    partner_indices = np.array(
        [idx for idx in range(1, n_levels) if idx != level_idx],
        dtype=int,
    )
    if partner_indices.size == 0:
        return "no candidate partners"

    energy_diffs = np.abs(tracked_evals[flux_idx, partner_indices] - branch_energy)
    parts = []
    for order_idx in np.argsort(energy_diffs)[:max_partners]:
        partner_idx = int(partner_indices[order_idx])
        key = _swap_key(level_idx, partner_idx, left_idx, right_idx)
        suffix = " blocked" if key in blocked_swap_keys else ""
        parts.append(
            f"level {partner_idx} gap={energy_diffs[order_idx]:.4g}{suffix}"
        )
    return "; ".join(parts)


def derivative_smoothness_post_pass(
    tracked_evals,
    phi_vals,
    threshold=DERIVATIVE_JUMP_WARN_THRESHOLD,
):
    smoothed_evals = np.array(tracked_evals, copy=True)
    swaps = []
    handled_jumps = set()
    swapped_windows = set()
    n_levels = smoothed_evals.shape[1]

    print("\nRunning symmetric derivative-smoothness post-pass...")
    print(
        "  "
        f"threshold={threshold:.3g} GHz/Phi0; checking positive-flux half "
        f"inside |Phi| <= {DERIVATIVE_CHECK_PHI_MAX:.3g}"
    )
    for level_idx in range(1, n_levels - 1):
        swaps_for_level = 0
        initial_jumps = _positive_half_derivative_jumps(
            smoothed_evals,
            phi_vals,
            level_idx,
            threshold=threshold,
        )
        if initial_jumps:
            preview = ", ".join(
                f"+{rec['phi']:.4f} ({rec['jump']:.3g})"
                for rec in initial_jumps[:4]
            )
            print(
                "  "
                f"level {level_idx}: {len(initial_jumps)} positive-half "
                f"jumps above threshold; worst {preview}"
            )
        while swaps_for_level < MAX_DERIVATIVE_SMOOTHING_SWAPS_PER_LEVEL:
            jumps = [
                jump
                for jump in _positive_half_derivative_jumps(
                    smoothed_evals,
                    phi_vals,
                    level_idx,
                    threshold=threshold,
                )
                if (level_idx, jump["flux_index"]) not in handled_jumps
            ]
            if not jumps:
                break

            jump = jumps[0]
            phi_abs = abs(jump["phi"])
            left_idx = int(np.argmin(np.abs(phi_vals + phi_abs)))
            right_idx = int(np.argmin(np.abs(phi_vals - phi_abs)))
            if left_idx > right_idx:
                left_idx, right_idx = right_idx, left_idx

            print(
                "    "
                f"checking level {level_idx} jump at +{phi_abs:.4f} Phi0 "
                f"(|Delta slope|={jump['jump']:.3g}, "
                f"before={jump['slope_before']:.3g}, "
                f"after={jump['slope_after']:.3g}); "
                f"window=[{phi_vals[left_idx]:+.4f}, {phi_vals[right_idx]:+.4f}]"
            )
            print(
                "    "
                "closest partners at jump: "
                + _partner_diagnostics(
                    smoothed_evals,
                    jump["flux_index"],
                    level_idx,
                    left_idx,
                    right_idx,
                    swapped_windows,
                )
            )
            print(
                "    "
                "closest local energy gaps: "
                + _partner_energy_gap_diagnostics(
                    smoothed_evals,
                    phi_vals,
                    level_idx,
                    jump["flux_index"],
                )
            )

            partner_idx = None
            closest_idx = None
            swap_left_idx = None
            swap_right_idx = None
            post_level_recheck = None
            post_partner_recheck = None
            accepted_evals = None
            partner_candidates = []
            for candidate_partner in range(1, n_levels):
                if candidate_partner == level_idx:
                    continue
                candidate_closest_idx = _closest_energy_index_near_jump(
                    smoothed_evals,
                    phi_vals,
                    level_idx,
                    candidate_partner,
                    jump["flux_index"],
                )
                candidate_phi_abs = abs(float(phi_vals[candidate_closest_idx]))
                candidate_left_idx = int(np.argmin(np.abs(phi_vals + candidate_phi_abs)))
                candidate_right_idx = int(np.argmin(np.abs(phi_vals - candidate_phi_abs)))
                if candidate_left_idx > candidate_right_idx:
                    candidate_left_idx, candidate_right_idx = (
                        candidate_right_idx,
                        candidate_left_idx,
                    )
                candidate_gap = abs(
                    smoothed_evals[candidate_closest_idx, candidate_partner]
                    - smoothed_evals[candidate_closest_idx, level_idx]
                )
                prefer_above_penalty = 0 if candidate_partner > level_idx else 1
                partner_candidates.append(
                    (
                        candidate_gap,
                        prefer_above_penalty,
                        abs(candidate_partner - level_idx),
                        candidate_partner,
                        candidate_closest_idx,
                        candidate_left_idx,
                        candidate_right_idx,
                    )
                )

            partner_candidates.sort(key=lambda item: item[:4])
            for (
                candidate_gap,
                _prefer_above_penalty,
                _branch_distance,
                candidate_partner,
                candidate_closest_idx,
                candidate_left_idx,
                candidate_right_idx,
            ) in partner_candidates:
                key = _swap_key(
                    level_idx,
                    candidate_partner,
                    candidate_left_idx,
                    candidate_right_idx,
                )
                if key in swapped_windows:
                    continue
                if candidate_gap > DERIVATIVE_MAX_CLOSEST_SWAP_GAP_GHZ:
                    break
                row_indices = np.arange(candidate_left_idx, candidate_right_idx + 1)
                trial_evals = _swap_columns_for_rows(
                    smoothed_evals,
                    row_indices,
                    level_idx,
                    candidate_partner,
                )
                trial_level_recheck = _derivative_jump_record_at(
                    trial_evals,
                    phi_vals,
                    jump["flux_index"],
                    level_idx,
                )
                trial_partner_recheck = _derivative_jump_record_at(
                    trial_evals,
                    phi_vals,
                    jump["flux_index"],
                    candidate_partner,
                )
                before_partner_recheck = _derivative_jump_record_at(
                    smoothed_evals,
                    phi_vals,
                    jump["flux_index"],
                    candidate_partner,
                )
                before_worst = max(
                    jump["jump"],
                    0.0
                    if before_partner_recheck is None
                    else before_partner_recheck["jump"],
                )
                after_worst = max(
                    np.inf
                    if trial_level_recheck is None
                    else trial_level_recheck["jump"],
                    np.inf
                    if trial_partner_recheck is None
                    else trial_partner_recheck["jump"],
                )
                improvement = (
                    jump["jump"]
                    - (
                        np.inf
                        if trial_level_recheck is None
                        else trial_level_recheck["jump"]
                    )
                )
                print(
                    "    "
                    f"candidate level {candidate_partner}: closest gap="
                    f"{candidate_gap:.4g} at {phi_vals[candidate_closest_idx]:+.4f}, "
                    f"post fitted jumps level {level_idx}="
                    f"{_format_jump_record(trial_level_recheck)}, "
                    f"level {candidate_partner}="
                    f"{_format_jump_record(trial_partner_recheck)}"
                )
                if (
                    improvement >= DERIVATIVE_REPAIR_MIN_TARGET_IMPROVEMENT
                    and after_worst <= before_worst
                ):
                    partner_idx = candidate_partner
                    closest_idx = candidate_closest_idx
                    swap_left_idx = candidate_left_idx
                    swap_right_idx = candidate_right_idx
                    accepted_evals = trial_evals
                    post_level_recheck = trial_level_recheck
                    post_partner_recheck = trial_partner_recheck
                    break

            if partner_idx is None:
                if partner_candidates:
                    best_gap, _, _, best_partner, best_idx, _, _ = partner_candidates[0]
                    print(
                        "  "
                        f"level {level_idx}: no local branch crossing close enough "
                        f"for jump at +{phi_abs:.4f} Phi0. Best candidate was "
                        f"level {best_partner} with min gap={best_gap:.4g} GHz "
                        f"at phi={phi_vals[best_idx]:+.4f}; require <= "
                        f"{DERIVATIVE_MAX_CLOSEST_SWAP_GAP_GHZ:.3g} GHz."
                    )
                else:
                    print(
                        "  "
                        f"level {level_idx}: no candidate partners for "
                        f"jump at +{phi_abs:.4f} Phi0."
                    )
                print(
                    "  "
                    f"level {level_idx}: no unswapped close partner remains for "
                    f"jump at +{phi_abs:.4f} Phi0; leaving it for review."
                )
                handled_jumps.add((level_idx, jump["flux_index"]))
                continue

            energy_gap = abs(
                smoothed_evals[jump["flux_index"], partner_idx]
                - smoothed_evals[jump["flux_index"], level_idx]
            )
            closest_gap = abs(
                smoothed_evals[closest_idx, partner_idx]
                - smoothed_evals[closest_idx, level_idx]
            )
            smoothed_evals = accepted_evals
            key = _swap_key(level_idx, partner_idx, swap_left_idx, swap_right_idx)
            swapped_windows.add(key)

            swaps_for_level += 1
            swap = {
                "level": int(level_idx),
                "partner": int(partner_idx),
                "phi": float(phi_abs),
                "closest_phi": float(phi_vals[closest_idx]),
                "left_phi": float(phi_vals[swap_left_idx]),
                "right_phi": float(phi_vals[swap_right_idx]),
                "interval": "closest-energy symmetric",
                "jump": float(jump["jump"]),
                "rechecked_jump": float(post_level_recheck["jump"]),
                "partner_rechecked_jump": float(post_partner_recheck["jump"]),
            }
            swaps.append(swap)

            for handled_level in (level_idx, partner_idx):
                for handled_flux in (
                    jump["flux_index"] - 1,
                    jump["flux_index"],
                    jump["flux_index"] + 1,
                    closest_idx - 1,
                    closest_idx,
                    closest_idx + 1,
                ):
                    if 0 <= handled_flux < len(phi_vals):
                        handled_jumps.add((handled_level, handled_flux))

            print(
                "  "
                f"level {level_idx} jump at +{phi_abs:.4f} Phi0 "
                f"(|Delta slope|={jump['jump']:.3g}) -> swapped with "
                f"level {partner_idx} over closest-energy symmetric window "
                f"[{phi_vals[swap_left_idx]:+.4f}, "
                f"{phi_vals[swap_right_idx]:+.4f}]. "
                f"gap at jump={energy_gap:.4g} GHz, "
                f"closest gap={closest_gap:.4g} GHz at "
                f"phi={phi_vals[closest_idx]:+.4f}; "
                f"re-check level {level_idx}={post_level_recheck['jump']:.3g}, "
                f"level {partner_idx}={post_partner_recheck['jump']:.3g}"
            )

        if swaps_for_level >= MAX_DERIVATIVE_SMOOTHING_SWAPS_PER_LEVEL:
            print(
                "  "
                f"level {level_idx}: stopped after "
                f"{MAX_DERIVATIVE_SMOOTHING_SWAPS_PER_LEVEL} swaps; "
                "remaining jumps may need manual review."
            )

    if not swaps:
        print("Derivative post-pass: no symmetric branch swaps needed.")
    else:
        print(f"Derivative post-pass: applied {len(swaps)} symmetric branch swaps.")
    return smoothed_evals, swaps


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
    report_derivative_jump_scale(tracked_evals, phi_vals)
    report_derivative_jumps(tracked_evals, phi_vals)

    smoothed_evals, _ = derivative_smoothness_post_pass(
        tracked_evals,
        phi_vals,
    )
    report_derivative_jumps(smoothed_evals, phi_vals)

    print("\nComputing dispersive coupling chi from dressed-energy double difference")
    chi_fit = chi_from_dressed_double_difference(
        smoothed_evals[mid_idx], phi_sweet=phi_vals[mid_idx]
    )
    print(f"chi / 2pi = {chi_fit * 1e3:.2f} MHz")

    print("Assigning |nq, nr> labels at flux points")
    level_labels_flux = label_flux_points(smoothed_evals, phi_vals, chi_fit)
    branch_labels = sweet_spot_branch_labels(level_labels_flux, mid_idx)
    print(
        f"Labels assigned only for |Phi| <= {LABEL_VALID_PHI_MAX:.2f} Phi0; "
        "edge labels are treated as unreliable."
    )

    print(
        f"\nPost-pass state labels at Phi = {phi_vals[mid_idx]:.3f} Phi0 "
        "(sweet spot):"
    )
    for level_idx in range(1, N_LEVELS):
        nq, nr = branch_labels.get(level_idx) or (-1, -1)
        print(
            f"tracked level {level_idx:2d} -> |{nq},{nr}>, "
            f"E = {smoothed_evals[mid_idx, level_idx]:.4f} GHz"
        )

    # Plot raw, tracked, and derivative-smoothed sweeps side by side.
    cmap = plt.cm.tab20
    colors = [cmap(i / max(N_LEVELS - 2, 1)) for i in range(N_LEVELS - 1)]

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    ax = axes[0]
    for i, level_idx in enumerate(range(1, N_LEVELS)):
        ax.plot(phi_vals, raw_evals[:, level_idx], color=colors[i], linewidth=1.2, alpha=0.8)
    ax.set_xlabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Energy (GHz)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Raw eigenvalues (no tracking)", fontsize=TITLE_FONTSIZE)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    ax = axes[1]
    for i, level_idx in enumerate(range(1, N_LEVELS)):
        label = branch_labels.get(level_idx)
        legend_label = (
            f"branch {level_idx}: unlabeled"
            if label is None
            else f"branch {level_idx}: |{label[0]},{label[1]}>"
        )
        ax.plot(phi_vals, tracked_evals[:, level_idx], color=colors[i], linewidth=1.2,
                alpha=0.8, label=legend_label)
    ax.set_xlabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Energy (GHz)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Tracked eigenvalues", fontsize=TITLE_FONTSIZE)
    ax.legend(frameon=False, ncols=2, fontsize=LEGEND_FONTSIZE, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    ax = axes[2]
    for i, level_idx in enumerate(range(1, N_LEVELS)):
        label = branch_labels.get(level_idx)
        legend_label = (
            f"branch {level_idx}: unlabeled"
            if label is None
            else f"branch {level_idx}: |{label[0]},{label[1]}>"
        )
        ax.plot(
            phi_vals,
            smoothed_evals[:, level_idx],
            color=colors[i],
            linewidth=1.2,
            alpha=0.8,
            label=legend_label,
        )
    ax.set_xlabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Energy (GHz)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Derivative-smoothed eigenvalues", fontsize=TITLE_FONTSIZE)
    ax.legend(frameon=False, ncols=2, fontsize=LEGEND_FONTSIZE, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    ymax = max(axis.get_ylim()[1] for axis in axes)
    for axis in axes:
        axis.set_ylim(-0.3, ymax)

    fig.suptitle(
        "Raw vs tracked vs derivative-smoothed eigenvalues",
        fontsize=SUPTITLE_FONTSIZE,
        y=0.93,
    )
    fig.tight_layout()
    print_model_summary(chi_fit)

    os.makedirs("plot_output", exist_ok=True)
    out_path = "plot_output/imet_energy_levels_heatmap_tracked_labelled.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

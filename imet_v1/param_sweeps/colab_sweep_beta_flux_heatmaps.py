"""
Standalone iMET beta/flux heatmap sweep script.

Colab setup (run once in a notebook cell before executing this script):
    !pip install -q scqubits scipy matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import scqubits as scq
import sys
import os
import csv
import time
from scipy.optimize import linear_sum_assignment

L_r_orig = 0.50e-9
C_r = 0.80e-12
L_c_orig = 0.16e-9
L_J1 = 30.0e-9
L_J2 = 30.0e-9
C_J1 = 40e-15
C_J2 = 40e-15

L_tot = L_r_orig + L_c_orig

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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
    E = np.asarray(E)
    k = int(np.argmin(np.abs(E - omega_r_hint)))
    return float(E[k]), k


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


def _nearest_level(E, target):
    idx = int(np.argmin(np.abs(E - target)))
    return idx, float(E[idx])


def _fit_low_level_pair_params(
    E,
    omega_q_hint=None,
    omega_r_hint=None,
    max_candidate_index=5,
):
    E = np.asarray(E, dtype=float)
    max_idx = min(len(E) - 1, max_candidate_index)
    if max_idx < 2:
        raise ValueError("Need at least two excited levels for low-level pair fit.")

    best = None
    for iq in range(1, max_idx + 1):
        for ir in range(1, max_idx + 1):
            if iq == ir:
                continue

            omega_q = float(E[iq])
            omega_r = float(E[ir])
            idx_20, E20 = _nearest_level(E, 2.0 * omega_q)
            idx_02, E02 = _nearest_level(E, 2.0 * omega_r)
            idx_11, E11 = _nearest_level(E, omega_q + omega_r)

            alpha_q = float(E20 - 2.0 * omega_q)
            alpha_r = float(E02 - 2.0 * omega_r)
            chi_qr = float(E11 - (omega_q + omega_r))
            resid_20 = abs(alpha_q)
            resid_02 = abs(alpha_r)
            resid_11 = abs(chi_qr)

            q_hint_miss = (
                abs(omega_q - float(omega_q_hint))
                if omega_q_hint is not None
                else 0.0
            )
            r_hint_miss = (
                abs(omega_r - float(omega_r_hint))
                if omega_r_hint is not None
                else 0.0
            )

            resonator_visible = 2.0 * omega_r <= float(E[-1]) + 0.25
            q_hint_tol = 1.25
            r_hint_tol = (
                max(0.6, 0.12 * float(omega_r_hint))
                if omega_r_hint is not None
                else np.inf
            )
            valid = (
                alpha_q < -0.02
                and abs(alpha_q) < 0.6
                and resid_20 < 0.35
                and resid_11 < 0.25
                and idx_20 not in {0, iq, ir}
                and idx_11 not in {0, iq, ir, idx_20}
            )
            if omega_q_hint is not None:
                valid = valid and q_hint_miss < q_hint_tol
            if omega_r_hint is not None:
                valid = valid and r_hint_miss < r_hint_tol
            if resonator_visible:
                valid = (
                    valid
                    and idx_02 not in {0, iq, ir, idx_20, idx_11}
                    and resid_02 < 0.12
                )
            if not valid:
                continue

            # Prefer a transmon-like qubit mode (negative anharmonicity),
            # a resonator-like mode (near-harmonic), and a small dispersive shift.
            score = 18.0 * resid_02 + 8.0 * resid_11 + 3.0 * resid_20

            if omega_q_hint is not None:
                score += 2.0 * q_hint_miss
            if omega_r_hint is not None:
                score += 1.5 * r_hint_miss

            if abs(alpha_q) <= abs(alpha_r):
                score += 30.0
            if iq > ir:
                score += 2.0
            if iq > 2:
                score += 2.0 * (iq - 2)
            if ir > 3:
                score += 1.0 * (ir - 3)

            candidate = dict(
                omega_q=omega_q,
                omega_r=omega_r,
                alpha_q=alpha_q,
                chi_qr=chi_qr,
                alpha_r=alpha_r,
                idx_q=iq,
                idx_r=ir,
                idx_20=idx_20,
                idx_02=idx_02,
                idx_11=idx_11,
                resid_20=resid_20,
                resid_02=resid_02,
                resid_11=resid_11,
                score=score,
            )
            if best is None or candidate["score"] < best["score"]:
                best = candidate

    if best is None:
        raise ValueError("Unable to fit low-level qubit/resonator pair.")
    return best


def fit_effective_params_2mode(
    evals_rel,
    tol_nonmultiple=0.20,
    omega_r_known=None,
    omega_r_hint=None,
    omega_q_hint=None,
):
    E = np.array(sorted(evals_rel))
    if E[0] != 0.0:
        E = E - E[0]

    try:
        params = _fit_low_level_pair_params(
            E,
            omega_q_hint=omega_q_hint,
            omega_r_hint=omega_r_hint,
        )
        # FIX: return ALL keys from _fit_low_level_pair_params, not just four.
        # fit_and_label_tracked_levels needs idx_q, idx_r, idx_20, idx_11, idx_02.
        return params
    except Exception:
        if omega_q_hint is not None or omega_r_hint is not None:
            raise

    if omega_r_known is not None:
        omega_r = float(omega_r_known)
    elif omega_r_hint is not None:
        omega_r, _ = _identify_omega_from_hint(E, float(omega_r_hint))
    else:
        omega_r = _peak_spacing_from_pairwise_diffs(
            E[: min(len(E), 16)], min_diff=1.0, bin_width=0.02
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
            omega_q = float(E[2]) if len(E) > 2 else float(E[1])

    idx_q = int(np.argmin(np.abs(E - omega_q)))
    idx_r = int(np.argmin(np.abs(E - omega_r)))

    target_2q = 2.0 * omega_q
    idx_20 = int(np.argmin(np.abs(E - target_2q)))
    alpha_q = float(E[idx_20] - target_2q)

    target_2r = 2.0 * omega_r
    idx_02 = int(np.argmin(np.abs(E - target_2r)))

    target_11 = omega_q + omega_r
    idx_11 = int(np.argmin(np.abs(E - target_11)))
    chi_qr = float(E[idx_11] - target_11)

    # FIX: return the same key set as _fit_low_level_pair_params so the caller
    # can access idx_q, idx_r, etc. without KeyError.
    return dict(
        omega_r=omega_r,
        omega_q=omega_q,
        alpha_q=alpha_q,
        chi_qr=chi_qr,
        alpha_r=float(E[idx_02] - target_2r),
        idx_q=idx_q,
        idx_r=idx_r,
        idx_20=idx_20,
        idx_02=idx_02,
        idx_11=idx_11,
        resid_20=abs(alpha_q),
        resid_02=abs(float(E[idx_02] - target_2r)),
        resid_11=abs(chi_qr),
        score=0.0,
    )


def predicted_energy_2mode(nq, nr, p, include_chi=True):
    omega_q, omega_r = p["omega_q"], p["omega_r"]
    alpha_q = p["alpha_q"]
    chi_qr = p["chi_qr"] if include_chi else 0.0
    return (
        omega_q * nq
        + 0.5 * alpha_q * nq * (nq - 1)
        + omega_r * nr
        + chi_qr * nq * nr
    )


def assign_labels_2mode(evals_rel, p, nq_max=6, nr_max=8, include_chi=True):
    E = np.array(sorted(evals_rel))
    E = E - E[0]

    cand = []
    for nq in range(nq_max + 1):
        for nr in range(nr_max + 1):
            Ep = predicted_energy_2mode(nq, nr, p, include_chi=include_chi)
            cand.append((Ep, nq, nr))
    cand.sort(key=lambda t: t[0])

    e_max = float(E[-1])
    anchor_specs = [(0.0, 0, 0)]
    for nq in range(1, nq_max + 1):
        target = (
            p["omega_q"] * nq
            + 0.5 * p["alpha_q"] * nq * (nq - 1)
        )
        if target <= e_max + 1.0:
            anchor_specs.append((target, nq, 0))
    for nr in range(1, nr_max + 1):
        target = p["omega_r"] * nr
        if target <= e_max + 1.0:
            anchor_specs.append((target, 0, nr))
    anchor_specs.sort(key=lambda t: t[0])

    used = set()
    used_energy_indices = set()
    anchored_by_index = {}
    for target, nq, nr in anchor_specs:
        idx = _match_nearest_unused_level(E, target, used_energy_indices)
        anchored_by_index[idx] = dict(
            k=idx,
            E=E[idx],
            nq=nq,
            nr=nr,
            E_pred=predicted_energy_2mode(nq, nr, p, include_chi=include_chi),
            resid=abs(E[idx] - target),
        )
        used.add((nq, nr))
        used_energy_indices.add(idx)

    out = []
    for k, Ek in enumerate(E):
        if k in anchored_by_index:
            out.append(anchored_by_index[k])
            continue

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


def energy_of_label(assignments, nq, nr):
    for assignment in assignments:
        if assignment["nq"] == nq and assignment["nr"] == nr:
            return assignment["E"]
    return None


def extract_exact_low_level_observables(assignments):
    E00 = energy_of_label(assignments, 0, 0)
    E10 = energy_of_label(assignments, 1, 0)
    E20 = energy_of_label(assignments, 2, 0)
    E01 = energy_of_label(assignments, 0, 1)
    E11 = energy_of_label(assignments, 1, 1)

    alpha_q = np.nan
    chi_qr = np.nan

    if E00 is not None and E10 is not None and E20 is not None:
        alpha_q = float((E20 - E10) - (E10 - E00))

    if E00 is not None and E10 is not None and E01 is not None and E11 is not None:
        chi_qr = float((E11 - E01) - (E10 - E00))

    return alpha_q, chi_qr


def omega_q_sweet_hint_ghz() -> float:
    E_J1_eff = E_J1
    E_J2_eff = E_J2
    E_J_sum = E_J1_eff + E_J2_eff
    E_C_eff = charging_energy_ghz(C_J1 + C_J2)
    if E_J_sum <= 0.0:
        return E_C_eff
    return float(np.sqrt(8.0 * E_J_sum * E_C_eff) - E_C_eff)


def transmon_energy_levels_ghz(E_J: float, E_C: float, n_max: int = 6) -> np.ndarray:
    if E_J <= 0.0:
        n = np.arange(n_max + 1, dtype=float)
        return n * E_C
    n = np.arange(n_max + 1, dtype=float)
    E_n = (
        np.sqrt(8.0 * E_J * E_C) * (n + 0.5)
        - (E_C / 12.0) * (6 * n**2 + 6 * n + 3)
        - E_J
    )
    return E_n - E_n[0]

PHI0 = 2.067833848e-15
E_CHARGE = 1.602176634e-19
H = 6.62607015e-34
MAX_OVERLAY_GAP_GHZ = 0.1
DERIVATIVE_JUMP_WARN_THRESHOLD = 5.0
LABEL_VALID_PHI_MAX = 0.45
LABEL_VALID_PHI_TOL = 1e-12
DERIVATIVE_CHECK_PHI_MAX = 0.45
MAX_DERIVATIVE_SMOOTHING_SWAPS_PER_LEVEL = 4 * 12
DERIVATIVE_POST_PASS_DIAGNOSTIC_PARTNERS = 4
DERIVATIVE_CLOSEST_ENERGY_SEARCH_RADIUS = 3
DERIVATIVE_MAX_CLOSEST_SWAP_GAP_GHZ = 0.75
DERIVATIVE_SMOOTHNESS_NEIGHBORHOOD = 3
DERIVATIVE_REPAIR_MIN_TARGET_IMPROVEMENT = 1.0
CHI_WARN_ABS_GHZ = 0.1
CHI_DISPLAY_ABS_MAX_GHZ = 0.1


def inductive_energy_ghz(L_H: float) -> float:
    EJ_J = (PHI0 / (2 * np.pi)) ** 2 / L_H
    return EJ_J / (H * 1e9)


def charging_energy_ghz(C_F: float) -> float:
    EC_J = E_CHARGE**2 / (2 * C_F)
    return EC_J / (H * 1e9)


E_J1 = inductive_energy_ghz(L_J1)
E_J2 = inductive_energy_ghz(L_J2)
E_C1 = charging_energy_ghz(C_J1)
E_C2 = charging_energy_ghz(C_J2)
E_C_r = charging_energy_ghz(C_r)


def build_circuit(L_c: float):
    if L_c < 1e-15:
        L_c = 1e-15
    L_r = L_tot - L_c
    if L_r < 1e-15:
        L_r = 1e-15

    E_L_c = inductive_energy_ghz(L_c)
    E_L_r = inductive_energy_ghz(L_r)

    iMET_yaml = f"""# iMET
    branches:
    - ["JJ", 1,4, {E_J1:.6g}, {E_C1:.6g}]
    - ["JJ", 1,2, {E_J2:.6g}, {E_C2:.6g}]
    - ["L", 2,4, {E_L_c:.6g}]
    - ["L", 2,3, {E_L_r:.6g}]
    - ["C", 3,4, {E_C_r:.6g}]
    """

    with HiddenPrints():
        circ = scq.Circuit(iMET_yaml, from_file=False, ext_basis="harmonic")
        circ.cutoff_n_1 = 6
        circ.cutoff_ext_2 = 10
        circ.cutoff_ext_3 = 10

    flux_syms = getattr(circ, "external_fluxes", None)
    if flux_syms is None or len(flux_syms) == 0:
        raise RuntimeError("No external flux variables found for iMET circuit.")

    return circ, str(flux_syms[0]), L_r


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
    prefix="",
    threshold=DERIVATIVE_JUMP_WARN_THRESHOLD,
    phi_max=DERIVATIVE_CHECK_PHI_MAX,
    max_reports=5,
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
        return []

    print(
        f"{prefix} derivative smoothness: {len(records)} fitted-slope jumps "
        f"above {threshold:.3g} GHz/Phi0 inside |Phi| <= {phi_max:.3g} "
        f"using {DERIVATIVE_SMOOTHNESS_NEIGHBORHOOD}-step neighborhoods"
    )
    for rec in records[:max_reports]:
        print(
            f"{prefix}   level={rec['level']:2d}, "
            f"phi={rec['phi']:+.4f}, "
            f"|Delta slope|={rec['jump']:.3g} GHz/Phi0, "
            f"before={rec['slope_before']:.3g}, "
            f"after={rec['slope_after']:.3g}"
        )
    if len(records) > max_reports:
        print(f"{prefix}   ... {len(records) - max_reports} more not shown")
    return records


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
    return {
        "level": int(level_idx),
        "flux_index": int(flux_index),
        "phi": float(phi_vals[flux_index]),
        "jump": float(abs(slope_after - slope_before)),
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


def _swap_columns_for_rows(evals, row_indices, level_idx, partner_idx):
    swapped = np.array(evals, copy=True)
    level_values = swapped[row_indices, level_idx].copy()
    swapped[row_indices, level_idx] = swapped[row_indices, partner_idx]
    swapped[row_indices, partner_idx] = level_values
    return swapped


def _partner_diagnostics(
    tracked_evals,
    phi_vals,
    flux_idx,
    level_idx,
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
        gap = abs(
            tracked_evals[closest_idx, partner_idx]
            - tracked_evals[closest_idx, level_idx]
        )
        records.append((gap, partner_idx, closest_idx))
    records.sort(key=lambda item: item[0])
    return "; ".join(
        f"level {partner_idx} min_gap={gap:.4g} at {phi_vals[closest_idx]:+.4f}"
        for gap, partner_idx, closest_idx in records[:max_partners]
    )


def derivative_smoothness_post_pass(
    tracked_evals,
    phi_vals,
    prefix="",
    threshold=DERIVATIVE_JUMP_WARN_THRESHOLD,
):
    smoothed_evals = np.array(tracked_evals, copy=True)
    swaps = []
    handled_jumps = set()
    swapped_windows = set()
    n_levels = smoothed_evals.shape[1]
    max_swaps_per_level = 4 * n_levels

    print(
        f"{prefix} running fitted-derivative branch post-pass "
        f"(threshold={threshold:.3g} GHz/Phi0, "
        f"|Phi| <= {DERIVATIVE_CHECK_PHI_MAX:.3g})"
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
                f"{prefix}   level {level_idx}: {len(initial_jumps)} "
                f"positive-half fitted jumps; worst {preview}"
            )

        while swaps_for_level < max_swaps_per_level:
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
            print(
                f"{prefix}     checking level {level_idx} jump at "
                f"+{phi_abs:.4f}: |Delta fitted slope|={jump['jump']:.3g}, "
                f"before={jump['slope_before']:.3g}, "
                f"after={jump['slope_after']:.3g}"
            )
            print(
                f"{prefix}     closest local energy gaps: "
                + _partner_diagnostics(
                    smoothed_evals,
                    phi_vals,
                    jump["flux_index"],
                    level_idx,
                )
            )

            accepted = None
            partner_candidates = []
            for candidate_partner in range(1, n_levels):
                if candidate_partner == level_idx:
                    continue
                closest_idx = _closest_energy_index_near_jump(
                    smoothed_evals,
                    phi_vals,
                    level_idx,
                    candidate_partner,
                    jump["flux_index"],
                )
                gap = abs(
                    smoothed_evals[closest_idx, candidate_partner]
                    - smoothed_evals[closest_idx, level_idx]
                )
                phi_center = abs(float(phi_vals[closest_idx]))
                left_idx = int(np.argmin(np.abs(phi_vals + phi_center)))
                right_idx = int(np.argmin(np.abs(phi_vals - phi_center)))
                if left_idx > right_idx:
                    left_idx, right_idx = right_idx, left_idx
                prefer_above_penalty = 0 if candidate_partner > level_idx else 1
                partner_candidates.append(
                    (
                        gap,
                        prefer_above_penalty,
                        abs(candidate_partner - level_idx),
                        candidate_partner,
                        closest_idx,
                        left_idx,
                        right_idx,
                    )
                )

            partner_candidates.sort(key=lambda item: item[:4])
            for (
                gap,
                _prefer_above_penalty,
                _branch_distance,
                candidate_partner,
                closest_idx,
                left_idx,
                right_idx,
            ) in partner_candidates:
                pair = tuple(sorted((int(level_idx), int(candidate_partner))))
                key = _swap_key(level_idx, candidate_partner, left_idx, right_idx)
                _pre_accepted = None
                _blocking_key = None
                if key in swapped_windows:
                    _blocking_key = key
                else:
                    for _sw_key in swapped_windows:
                        if _sw_key[:2] == pair:
                            _sw_lo, _sw_hi = _sw_key[2], _sw_key[3]
                            if _sw_lo <= right_idx and _sw_hi >= left_idx:
                                _blocking_key = _sw_key
                                break
                if _blocking_key is not None:
                    # The proposed window overlaps an already-swapped window for
                    # this pair. Try centering on the jump, then iteratively
                    # expand outward until the window boundary lies PAST the
                    # crossing region (or hits phi_max).
                    blocked_key = _blocking_key
                    blocked_lo, blocked_hi = blocked_key[2], blocked_key[3]
                    jump_phi_center = abs(float(phi_vals[jump["flux_index"]]))
                    _ext_left = int(np.argmin(np.abs(phi_vals + jump_phi_center)))
                    _ext_right = int(np.argmin(np.abs(phi_vals - jump_phi_center)))
                    if _ext_left > _ext_right:
                        _ext_left, _ext_right = _ext_right, _ext_left

                    while True:
                        alt_key = _swap_key(
                            level_idx, candidate_partner, _ext_left, _ext_right
                        )
                        if alt_key in swapped_windows or alt_key == blocked_key:
                            _ext_left = max(0, _ext_left - 2)
                            _ext_right = min(len(phi_vals) - 1, _ext_right + 2)
                            continue
                        if abs(phi_vals[_ext_right]) > DERIVATIVE_CHECK_PHI_MAX:
                            break
                        alt_gap = abs(
                            smoothed_evals[jump["flux_index"], candidate_partner]
                            - smoothed_evals[jump["flux_index"], level_idx]
                        )
                        if alt_gap > DERIVATIVE_MAX_CLOSEST_SWAP_GAP_GHZ:
                            break
                        # Only swap extension rows, excluding the already-swapped
                        # blocked window to avoid double-swapping (which un-swaps).
                        _ext_row_indices = np.array(
                            [i for i in range(_ext_left, _ext_right + 1)
                             if i < blocked_lo or i > blocked_hi],
                            dtype=int,
                        )
                        _ext_trial_evals = _swap_columns_for_rows(
                            smoothed_evals,
                            _ext_row_indices,
                            level_idx,
                            candidate_partner,
                        )
                        _ext_level_recheck = _derivative_jump_record_at(
                            _ext_trial_evals, phi_vals, jump["flux_index"], level_idx
                        )
                        _ext_partner_recheck = _derivative_jump_record_at(
                            _ext_trial_evals,
                            phi_vals,
                            jump["flux_index"],
                            candidate_partner,
                        )
                        _ext_before_partner = _derivative_jump_record_at(
                            smoothed_evals,
                            phi_vals,
                            jump["flux_index"],
                            candidate_partner,
                        )
                        _ext_before_worst = max(
                            jump["jump"],
                            0.0
                            if _ext_before_partner is None
                            else _ext_before_partner["jump"],
                        )
                        _ext_after_worst = max(
                            np.inf
                            if _ext_level_recheck is None
                            else _ext_level_recheck["jump"],
                            np.inf
                            if _ext_partner_recheck is None
                            else _ext_partner_recheck["jump"],
                        )
                        _ext_improvement = jump["jump"] - (
                            np.inf
                            if _ext_level_recheck is None
                            else _ext_level_recheck["jump"]
                        )
                        _ext_target_fixed = (
                            _ext_level_recheck is None
                            or _ext_level_recheck["jump"] < threshold
                        )
                        print(
                            f"{prefix}     candidate level {candidate_partner}: "
                            f"extension window "
                            f"[{phi_vals[_ext_left]:+.4f}, {phi_vals[_ext_right]:+.4f}] "
                            f"(excl [{phi_vals[blocked_lo]:+.4f}, "
                            f"{phi_vals[blocked_hi]:+.4f}]) "
                            f"gap={alt_gap:.4g}, "
                            f"post jumps level {level_idx}="
                            f"{_format_jump_record(_ext_level_recheck)}, "
                            f"level {candidate_partner}="
                            f"{_format_jump_record(_ext_partner_recheck)}"
                        )
                        if (
                            _ext_improvement >= DERIVATIVE_REPAIR_MIN_TARGET_IMPROVEMENT
                            and (
                                _ext_after_worst <= _ext_before_worst
                                or _ext_target_fixed
                            )
                        ):
                            _pre_accepted = {
                                "evals": _ext_trial_evals,
                                "partner": candidate_partner,
                                "closest_idx": jump["flux_index"],
                                "left_idx": _ext_left,
                                "right_idx": _ext_right,
                                "gap": alt_gap,
                                "level_recheck": _ext_level_recheck,
                                "partner_recheck": _ext_partner_recheck,
                            }
                            break
                        # Not improved enough — expand window by 2 flux steps each side
                        _ext_left = max(0, _ext_left - 2)
                        _ext_right = min(len(phi_vals) - 1, _ext_right + 2)

                    if _pre_accepted is None:
                        continue  # No viable extension; try next candidate_partner

                    accepted = _pre_accepted
                    break  # Extension fix accepted; done with partner_candidates

                if gap > DERIVATIVE_MAX_CLOSEST_SWAP_GAP_GHZ:
                    break

                row_indices = np.arange(left_idx, right_idx + 1)
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
                    f"{prefix}     candidate level {candidate_partner}: "
                    f"closest gap={gap:.4g} at {phi_vals[closest_idx]:+.4f}, "
                    f"post jumps level {level_idx}="
                    f"{_format_jump_record(trial_level_recheck)}, "
                    f"level {candidate_partner}="
                    f"{_format_jump_record(trial_partner_recheck)}"
                )
                target_fixed = (
                    trial_level_recheck is None
                    or trial_level_recheck["jump"] < threshold
                )
                if (
                    improvement >= DERIVATIVE_REPAIR_MIN_TARGET_IMPROVEMENT
                    and (after_worst <= before_worst or target_fixed)
                ):
                    accepted = {
                        "evals": trial_evals,
                        "partner": candidate_partner,
                        "closest_idx": closest_idx,
                        "left_idx": left_idx,
                        "right_idx": right_idx,
                        "gap": gap,
                        "level_recheck": trial_level_recheck,
                        "partner_recheck": trial_partner_recheck,
                    }
                    break

            if accepted is None:
                if partner_candidates:
                    best_gap, _, _, best_partner, best_idx, _, _ = partner_candidates[0]
                    print(
                        f"{prefix}   level {level_idx}: no accepted swap for "
                        f"+{phi_abs:.4f}; best local gap was level "
                        f"{best_partner} gap={best_gap:.4g} at "
                        f"{phi_vals[best_idx]:+.4f}"
                    )
                handled_jumps.add((level_idx, jump["flux_index"]))
                continue

            partner_idx = accepted["partner"]
            smoothed_evals = accepted["evals"]
            swapped_windows.add(
                _swap_key(
                    level_idx,
                    partner_idx,
                    accepted["left_idx"],
                    accepted["right_idx"],
                )
            )
            swaps_for_level += 1

            for handled_level in (level_idx, partner_idx):
                for handled_flux in (
                    jump["flux_index"] - 1,
                    jump["flux_index"],
                    jump["flux_index"] + 1,
                    accepted["closest_idx"] - 1,
                    accepted["closest_idx"],
                    accepted["closest_idx"] + 1,
                ):
                    if 0 <= handled_flux < len(phi_vals):
                        handled_jumps.add((handled_level, handled_flux))

            swap = {
                "level": int(level_idx),
                "partner": int(partner_idx),
                "phi": float(phi_abs),
                "closest_phi": float(phi_vals[accepted["closest_idx"]]),
                "left_phi": float(phi_vals[accepted["left_idx"]]),
                "right_phi": float(phi_vals[accepted["right_idx"]]),
                "jump": float(jump["jump"]),
                "rechecked_jump": float(accepted["level_recheck"]["jump"])
                if accepted["level_recheck"] is not None
                else 0.0,
                "partner_rechecked_jump": float(accepted["partner_recheck"]["jump"])
                if accepted["partner_recheck"] is not None
                else 0.0,
            }
            swaps.append(swap)
            print(
                f"{prefix}   level {level_idx} jump at +{phi_abs:.4f} "
                f"-> swapped with level {partner_idx} over "
                f"[{swap['left_phi']:+.4f}, {swap['right_phi']:+.4f}], "
                f"closest gap={accepted['gap']:.4g} GHz at "
                f"{swap['closest_phi']:+.4f}; post jumps "
                f"level {level_idx}={swap['rechecked_jump']:.3g}, "
                f"level {partner_idx}={swap['partner_rechecked_jump']:.3g}"
            )

        if swaps_for_level >= max_swaps_per_level:
            print(
                f"{prefix}   level {level_idx}: stopped after "
                f"{max_swaps_per_level} post-pass swaps"
            )

    if swaps:
        print(f"{prefix} derivative post-pass applied {len(swaps)} swaps")
    else:
        print(f"{prefix} derivative post-pass made no swaps")
    return smoothed_evals, swaps


def _candidate_chi_at_index(tracked_evals, flux_idx, label_indices):
    required = ((0, 0), (1, 0), (0, 1), (1, 1))
    if any(label not in label_indices for label in required):
        return np.nan
    e00 = tracked_evals[flux_idx, label_indices[(0, 0)]]
    e10 = tracked_evals[flux_idx, label_indices[(1, 0)]]
    e01 = tracked_evals[flux_idx, label_indices[(0, 1)]]
    e11 = tracked_evals[flux_idx, label_indices[(1, 1)]]
    return float(e11 - e10 - e01 + e00)


def _labels_from_indices(n_levels, label_indices):
    labels = [None] * n_levels
    for label, branch_idx in label_indices.items():
        if 0 <= branch_idx < n_levels and labels[branch_idx] is None:
            labels[branch_idx] = label
    return labels


def _fit_based_sweet_spot_candidate(evals_mid, omega_r_hint, omega_q_hint):
    params = fit_effective_params_2mode(
        evals_mid,
        omega_r_hint=omega_r_hint,
        omega_q_hint=omega_q_hint,
    )
    label_indices = {
        (0, 0): 0,
        (1, 0): int(params["idx_q"]),
        (0, 1): int(params["idx_r"]),
        (2, 0): int(params["idx_20"]),
        (1, 1): int(params["idx_11"]),
        (0, 2): int(params["idx_02"]),
    }
    if len(set(label_indices.values())) != len(label_indices):
        raise ValueError("Fit-based sweet-spot labels map multiple labels to one branch.")
    return label_indices


def _analytic_sweet_spot_candidate(evals_mid, omega_r_hint):
    energies = np.asarray(evals_mid, dtype=float)
    energies = energies - energies[0]
    e_q = transmon_energy_levels_ghz(
        E_J1 + E_J2,
        charging_energy_ghz(C_J1 + C_J2),
        n_max=6,
    )

    candidates = []
    for nq in range(7):
        for nr in range(8):
            candidates.append((e_q[nq] + nr * omega_r_hint, nq, nr))
    candidates.sort(key=lambda item: item[0])

    n_items = min(len(energies), len(candidates))
    cost = np.full((n_items, n_items), 1e9)
    for energy_idx in range(n_items):
        for cand_idx in range(n_items):
            cost[energy_idx, cand_idx] = abs(energies[energy_idx] - candidates[cand_idx][0])

    row_ind, col_ind = linear_sum_assignment(cost)
    label_indices = {}
    for energy_idx, cand_idx in zip(row_ind, col_ind):
        _, nq, nr = candidates[cand_idx]
        label = (nq, nr)
        if label not in label_indices:
            label_indices[label] = int(energy_idx)

    if (0, 0) not in label_indices:
        label_indices[(0, 0)] = 0
    return label_indices


def sweet_spot_branch_labels_and_indices(
    tracked_evals: np.ndarray,
    mid_idx: int,
    omega_r_hint: float,
    omega_q_hint: float | None,
    prefix: str = "",
):
    candidates = []
    evals_mid = tracked_evals[mid_idx]

    for name, builder in (
        (
            "fit",
            lambda: _fit_based_sweet_spot_candidate(
                evals_mid,
                omega_r_hint=omega_r_hint,
                omega_q_hint=omega_q_hint,
            ),
        ),
        ("analytic", lambda: _analytic_sweet_spot_candidate(evals_mid, omega_r_hint)),
    ):
        try:
            label_indices = builder()
            chi_mid = _candidate_chi_at_index(tracked_evals, mid_idx, label_indices)
            candidates.append((name, label_indices, chi_mid))
            chi_text = "nan" if not np.isfinite(chi_mid) else f"{chi_mid * 1e3:.6g} MHz"
            print(f"{prefix} sweet-spot {name} chi candidate: {chi_text}")
        except Exception as exc:
            print(f"{prefix} sweet-spot {name} labeling failed: {type(exc).__name__}: {exc}")

    sane = [
        candidate for candidate in candidates
        if np.isfinite(candidate[2]) and abs(candidate[2]) <= CHI_WARN_ABS_GHZ
    ]
    if sane:
        name, label_indices, chi_mid = min(sane, key=lambda item: abs(item[2]))
    elif candidates:
        name, label_indices, chi_mid = min(
            candidates,
            key=lambda item: abs(item[2]) if np.isfinite(item[2]) else np.inf,
        )
    else:
        raise ValueError("Unable to build any sweet-spot branch labels.")

    print(
        f"{prefix} using {name} sweet-spot labels; "
        f"chi_mid={chi_mid * 1e3:.6g} MHz"
    )
    for label in ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)):
        idx = label_indices.get(label)
        if idx is None:
            print(f"{prefix}   {format_label(label)} -> missing")
        else:
            print(
                f"{prefix}   {format_label(label)} -> branch {idx}, "
                f"E_mid={evals_mid[idx]:.9g} GHz"
            )

    display_label_indices = {}
    try:
        display_label_indices = _analytic_sweet_spot_candidate(evals_mid, omega_r_hint)
    except Exception:
        display_label_indices = {}

    labels = _labels_from_indices(tracked_evals.shape[1], display_label_indices)
    for label, idx in label_indices.items():
        if 0 <= idx < len(labels):
            labels = [None if existing == label else existing for existing in labels]
            labels[idx] = label

    labeled_count = sum(label is not None for label in labels)
    print(
        f"{prefix} display labels assigned to "
        f"{labeled_count}/{len(labels)} tracked branches"
    )
    return labels, label_indices


def branch_followed_observables(tracked_evals, label_indices):
    n_flux = tracked_evals.shape[0]
    alpha_col = np.full(n_flux, np.nan)
    chi_col = np.full(n_flux, np.nan)

    idx_00 = label_indices.get((0, 0))
    idx_10 = label_indices.get((1, 0))
    idx_20 = label_indices.get((2, 0))
    idx_01 = label_indices.get((0, 1))
    idx_11 = label_indices.get((1, 1))

    if idx_00 is None:
        idx_00 = 0
    e00 = tracked_evals[:, idx_00]

    if idx_10 is not None and idx_20 is not None:
        e10 = tracked_evals[:, idx_10]
        e20 = tracked_evals[:, idx_20]
        alpha_col = e20 - 2.0 * e10 + e00

    if idx_10 is not None and idx_01 is not None and idx_11 is not None:
        e10 = tracked_evals[:, idx_10]
        e01 = tracked_evals[:, idx_01]
        e11 = tracked_evals[:, idx_11]
        chi_col = e11 - e10 - e01 + e00

    return alpha_col, chi_col


def labeled_tracked_energy_table(tracked_evals, label_indices):
    state_order = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2))
    state_energies = np.full((len(state_order), tracked_evals.shape[0]), np.nan)
    state_branch_indices = {}

    for row_idx, label in enumerate(state_order):
        branch_idx = label_indices.get(label)
        if branch_idx is None:
            continue
        state_branch_indices[label] = int(branch_idx)
        state_energies[row_idx] = tracked_evals[:, branch_idx]

    return state_order, state_energies, state_branch_indices


def stabilize_pure_resonator_state_energies(
    state_order,
    state_energies,
    mid_idx,
    prefix="",
):
    stabilized_energies = np.array(state_energies, copy=True)
    pinned_states = []
    for row_idx, label in enumerate(state_order):
        nq, _ = label
        if nq != 0:
            continue
        sweet_spot_energy = stabilized_energies[row_idx, mid_idx]
        if not np.isfinite(sweet_spot_energy):
            continue
        stabilized_energies[row_idx, :] = sweet_spot_energy
        pinned_states.append((label, sweet_spot_energy))

    if pinned_states:
        pinned_text = ", ".join(
            f"{format_label(label)}={energy:.9g} GHz"
            for label, energy in pinned_states
        )
        print(
            f"{prefix} holding pure resonator state energies fixed at "
            f"sweet spot: {pinned_text}"
        )

    return stabilized_energies


def observables_from_labeled_tracked_energies(state_order, state_energies):
    row_for_label = {label: idx for idx, label in enumerate(state_order)}
    n_flux = state_energies.shape[1]
    alpha_col = np.full(n_flux, np.nan)
    chi_col = np.full(n_flux, np.nan)

    if all(label in row_for_label for label in ((0, 0), (1, 0), (2, 0))):
        e00 = state_energies[row_for_label[(0, 0)]]
        e10 = state_energies[row_for_label[(1, 0)]]
        e20 = state_energies[row_for_label[(2, 0)]]
        alpha_col = e20 - 2.0 * e10 + e00

    if all(label in row_for_label for label in ((0, 0), (1, 0), (0, 1), (1, 1))):
        e00 = state_energies[row_for_label[(0, 0)]]
        e10 = state_energies[row_for_label[(1, 0)]]
        e01 = state_energies[row_for_label[(0, 1)]]
        e11 = state_energies[row_for_label[(1, 1)]]
        chi_col = e11 - e10 - e01 + e00

    return alpha_col, chi_col


def fit_and_label_tracked_levels(
    evals_rel: np.ndarray,
    omega_r_hint: float,
    omega_q_hint: float | None = None,
):
    params = fit_effective_params_2mode(
        evals_rel,
        omega_r_hint=omega_r_hint,
        omega_q_hint=omega_q_hint,
    )
    try:
        assignments = assign_labels_2mode(evals_rel, params, include_chi=True)
    except Exception:
        assignments = []

    # Use exact low-level observables from assignments when available,
    # as these are more reliable than the fitted model parameters.
    alpha_exact, chi_exact = np.nan, np.nan
    if assignments:
        alpha_exact, chi_exact = extract_exact_low_level_observables(assignments)

    tracked_labels = [None] * len(evals_rel)
    tracked_labels[int(params["idx_q"])] = (1, 0)
    tracked_labels[int(params["idx_r"])] = (0, 1)
    tracked_labels[int(params["idx_20"])] = (2, 0)
    tracked_labels[int(params["idx_11"])] = (1, 1)
    tracked_labels[int(params["idx_02"])] = (0, 2)

    unmatched = [a for a in assignments if tracked_labels[a["k"]] is None]
    for energy_idx, energy in enumerate(evals_rel):
        if tracked_labels[energy_idx] is not None or not unmatched:
            continue
        best_idx = min(
            range(len(unmatched)),
            key=lambda idx: abs(unmatched[idx]["E"] - energy),
        )
        best = unmatched.pop(best_idx)
        tracked_labels[energy_idx] = (best["nq"], best["nr"])

    # Prefer the exact extraction; fall back to fitted params if exact is NaN.
    final_alpha = alpha_exact if np.isfinite(alpha_exact) else params["alpha_q"]
    final_chi = chi_exact if np.isfinite(chi_exact) else params["chi_qr"]

    return {
        "params": params,
        "assignments": assignments,
        "tracked_labels": tracked_labels,
        "alpha_q": final_alpha,
        "chi_qr": final_chi,
    }


def progress_stride(total_steps: int) -> int:
    return max(1, total_steps // 5)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {sec:02d}s"
    return f"{sec:d}s"


def sweep_beta_column(
    L_c: float,
    phi_vals: np.ndarray,
    evals_count: int,
    beta_index: int | None = None,
    beta_total: int | None = None,
):
    circ, flux_attr, L_r = build_circuit(L_c)
    omega_r_hint = 1.0 / (2 * np.pi * np.sqrt(L_tot * C_r)) / 1e9
    omega_q_hint = omega_q_sweet_hint_ghz()

    n_flux = len(phi_vals)
    mid_idx = n_flux // 2
    raw_evals = np.full((n_flux, evals_count), np.nan)
    all_evecs = []
    beta = L_c / L_tot
    prefix = (
        f"[beta {beta_index}/{beta_total} | beta={beta:.6f}]"
        if beta_index is not None and beta_total is not None
        else f"[beta={beta:.6f}]"
    )
    diag_stride = progress_stride(n_flux)
    beta_start = time.perf_counter()

    print(f"{prefix} starting tracked sweep across {n_flux} flux points")
    print(
        f"{prefix} hints: omega_r={omega_r_hint:.6f} GHz "
        f"(using L_tot={L_tot * 1e9:.6f} nH), "
        f"omega_q={omega_q_hint:.6f} GHz"
    )

    try:
        for idx, phi_ext in enumerate(phi_vals):
            with HiddenPrints():
                setattr(circ, flux_attr, float(phi_ext))
                evals, evecs = circ.eigensys(evals_count=evals_count)

            evals = np.asarray(evals, dtype=float)
            raw_evals[idx] = evals - evals[0]

            evecs = np.array(evecs)
            if evecs.ndim == 1:
                evecs = evecs[:, np.newaxis]
            if evecs.shape[0] < evecs.shape[1]:
                evecs = evecs.T
            all_evecs.append(evecs)

            step = idx + 1
            if step == 1 or step % diag_stride == 0 or step == n_flux:
                elapsed_beta = time.perf_counter() - beta_start
                avg_per_flux = elapsed_beta / step
                eta_beta = avg_per_flux * (n_flux - step)
                print(
                    f"{prefix} sweep {step}/{n_flux} "
                    f"(phi={phi_ext:.6f}, elapsed={format_duration(elapsed_beta)}, "
                    f"eta={format_duration(eta_beta)})"
                )
    except Exception:
        return None

    tracked_evals, _ = track_eigenbranches(raw_evals, all_evecs, phi_vals, mid_idx)
    report_derivative_jumps(tracked_evals, phi_vals, prefix=prefix)
    tracked_evals, _ = derivative_smoothness_post_pass(
        tracked_evals,
        phi_vals,
        prefix=prefix,
    )
    report_derivative_jumps(tracked_evals, phi_vals, prefix=prefix)

    tracked_labels = [[None] * evals_count for _ in range(n_flux)]

    try:
        branch_labels, label_indices = sweet_spot_branch_labels_and_indices(
            tracked_evals,
            mid_idx,
            omega_r_hint=omega_r_hint,
            omega_q_hint=omega_q_hint,
            prefix=prefix,
        )
    except Exception as exc:
        print(f"{prefix} sweet-spot branch labeling failed: {type(exc).__name__}: {exc}")
        branch_labels = [None] * evals_count
        label_indices = {}

    for idx, phi_ext in enumerate(phi_vals):
        if abs(phi_ext) <= LABEL_VALID_PHI_MAX + LABEL_VALID_PHI_TOL:
            tracked_labels[idx] = list(branch_labels)

    state_order, state_energies, state_branch_indices = labeled_tracked_energy_table(
        tracked_evals,
        label_indices,
    )
    state_energies = stabilize_pure_resonator_state_energies(
        state_order,
        state_energies,
        mid_idx,
        prefix=prefix,
    )
    alpha_col, chi_col = observables_from_labeled_tracked_energies(
        state_order,
        state_energies,
    )
    missing_labels = [
        format_label(label)
        for label in ((1, 0), (2, 0), (0, 1), (1, 1))
        if label not in label_indices
    ]
    if missing_labels:
        print(
            f"{prefix} branch-followed observables missing labels: "
            f"{', '.join(missing_labels)}"
        )
    print(f"{prefix} labeled tracked-energy rows used for heatmap observables:")
    for label in state_order:
        branch_idx = state_branch_indices.get(label)
        if branch_idx is None:
            print(f"{prefix}   {format_label(label)} -> missing")
        else:
            print(f"{prefix}   {format_label(label)} -> branch {branch_idx}")
    finite_chi = chi_col[np.isfinite(chi_col)]
    if finite_chi.size:
        warn_count = int(np.sum(np.abs(finite_chi) > CHI_WARN_ABS_GHZ))
        print(
            f"{prefix} tracked-energy heatmap chi range: "
            f"{np.nanmin(finite_chi) * 1e3:.6g} to "
            f"{np.nanmax(finite_chi) * 1e3:.6g} MHz; "
            f"chi(phi=0)={chi_col[mid_idx] * 1e3:.6g} MHz; "
            f"finite={finite_chi.size}/{len(chi_col)}; "
            f"|chi|>{CHI_WARN_ABS_GHZ * 1e3:.0f} MHz at {warn_count} points"
        )
        for sample_phi in (-0.25, 0.0, 0.25):
            sample_idx = int(np.argmin(np.abs(phi_vals - sample_phi)))
            value = chi_col[sample_idx]
            value_text = "nan" if not np.isfinite(value) else f"{value * 1e3:.6g} MHz"
            print(
                f"{prefix}   chi(phi={phi_vals[sample_idx]:+.3f}) = {value_text}"
            )
    else:
        print(f"{prefix} tracked-energy heatmap chi is all NaN")

    return {
        "alpha": alpha_col,
        "chi": chi_col,
        "tracked_evals": tracked_evals,
        "tracked_labels": tracked_labels,
        "state_order": state_order,
        "state_energies": state_energies,
    }


def format_label(label):
    if label is None:
        return "unlabeled"
    return f"|{label[0]},{label[1]}>"


def state_column_name(label):
    return f"state_{label[0]}_{label[1]}_GHz"


def write_sweep_csv(path, phi_vals, column_results):
    max_levels = max(
        (result["tracked_evals"].shape[1] for result in column_results if result is not None),
        default=0,
    )
    state_order = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2))
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["beta", "phi_ext_over_phi0", "alpha_GHz", "chi_GHz", "abs_chi_GHz"]
        header.extend(state_column_name(label) for label in state_order)
        header.extend(f"tracked_level_{level_idx}_GHz" for level_idx in range(max_levels))
        header.extend(f"tracked_level_{level_idx}_label" for level_idx in range(max_levels))
        writer.writerow(header)

        for result in column_results:
            if result is None:
                continue
            beta = result["beta"]
            tracked_evals = result["tracked_evals"]
            tracked_labels = result["tracked_labels"]
            state_energies = result.get("state_energies")
            alpha_col = result["alpha"]
            chi_col = result["chi"]
            for flux_idx, phi in enumerate(phi_vals):
                row = [
                    beta,
                    phi,
                    alpha_col[flux_idx],
                    chi_col[flux_idx],
                    abs(chi_col[flux_idx]) if np.isfinite(chi_col[flux_idx]) else np.nan,
                ]
                if state_energies is None:
                    row.extend(np.nan for _ in state_order)
                else:
                    row.extend(
                        state_energies[state_idx, flux_idx]
                        for state_idx in range(len(state_order))
                    )
                row.extend(tracked_evals[flux_idx, level_idx] for level_idx in range(tracked_evals.shape[1]))
                row.extend("" for _ in range(max_levels - tracked_evals.shape[1]))
                for level_idx in range(tracked_evals.shape[1]):
                    row.append(format_label(tracked_labels[flux_idx][level_idx]))
                row.extend("" for _ in range(max_levels - tracked_evals.shape[1]))
                writer.writerow(row)


def write_crossings_csv(path, crossing_records):
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "beta",
                "phi_ext_over_phi0",
                "kind",
                "level_a",
                "level_b",
                "label_a",
                "label_b",
                "gap_GHz",
            ]
        )
        for rec in crossing_records:
            writer.writerow(
                [
                    rec["beta"],
                    rec["phi"],
                    rec["kind"],
                    rec["level_a"],
                    rec["level_b"],
                    format_label(rec["label_a"]),
                    format_label(rec["label_b"]),
                    rec["gap_ghz"],
                ]
            )


def plot_beta_flux_sweeps(phi_vals, column_results):
    valid_columns = [result for result in column_results if result is not None]
    if not valid_columns:
        return None

    n_cols = 3
    n_rows = int(np.ceil(len(valid_columns) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 4.2 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    max_levels = max(result["tracked_evals"].shape[1] for result in valid_columns)
    plotted_levels = list(range(1, max_levels))
    cmap = plt.cm.tab20
    colors = {
        level_idx: cmap((level_idx - 1) / max(len(plotted_levels) - 1, 1))
        for level_idx in plotted_levels
    }

    for ax, result in zip(axes.flat, valid_columns):
        tracked_evals = result["tracked_evals"]
        tracked_labels = result["tracked_labels"]
        beta = result["beta"]
        mid_idx = len(phi_vals) // 2
        mid_labels = (
            tracked_labels[mid_idx]
            if mid_idx < len(tracked_labels)
            else [None] * tracked_evals.shape[1]
        )

        for level_idx in range(1, tracked_evals.shape[1]):
            label = mid_labels[level_idx] if level_idx < len(mid_labels) else None
            legend_label = (
                f"branch {level_idx}: unlabeled"
                if label is None
                else f"branch {level_idx}: {format_label(label)}"
            )
            ax.plot(
                phi_vals,
                tracked_evals[:, level_idx],
                color=colors[level_idx],
                linewidth=1.15,
                alpha=0.9,
                label=legend_label,
            )

        ax.axvline(-LABEL_VALID_PHI_MAX, color="0.65", linewidth=0.8, linestyle="--")
        ax.axvline(LABEL_VALID_PHI_MAX, color="0.65", linewidth=0.8, linestyle="--")
        ax.set_title(rf"$\beta={beta:.3f}$")
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=9)
        ax.legend(frameon=False, ncols=2, fontsize=6.5, loc="upper right")

    for ax in axes.flat[len(valid_columns):]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Tracked energy (GHz)")

    fig.suptitle("Tracked flux sweeps by beta", fontsize=14)
    fig.tight_layout()
    return fig


def detect_tracked_crossings(
    beta: float, phi_vals: np.ndarray, tracked_evals: np.ndarray, tracked_labels
):
    n_flux, n_levels = tracked_evals.shape
    if n_flux < 3:
        return []

    order = np.argsort(tracked_evals, axis=1)
    rank = np.empty_like(order)
    for idx in range(n_flux):
        rank[idx, order[idx]] = np.arange(n_levels)

    records = []

    def add_record(kind, phi_cross, gap_ghz, level_a, level_b, flux_idx):
        if not np.isfinite(gap_ghz) or gap_ghz >= MAX_OVERLAY_GAP_GHZ:
            return
        label_a = tracked_labels[flux_idx][level_a] if flux_idx < len(tracked_labels) else None
        label_b = tracked_labels[flux_idx][level_b] if flux_idx < len(tracked_labels) else None
        records.append(
            {
                "beta": beta,
                "phi": float(phi_cross),
                "gap_ghz": float(gap_ghz),
                "level_a": int(level_a),
                "level_b": int(level_b),
                "label_a": label_a,
                "label_b": label_b,
                "kind": kind,
            }
        )

    for level_a in range(1, n_levels):
        for level_b in range(level_a + 1, n_levels):
            diff = tracked_evals[:, level_a] - tracked_evals[:, level_b]
            gap = np.abs(diff)

            for idx in range(n_flux - 1):
                if not np.isfinite(diff[idx]) or not np.isfinite(diff[idx + 1]):
                    continue
                if diff[idx] == 0.0:
                    phi_cross = phi_vals[idx]
                elif diff[idx] * diff[idx + 1] > 0.0:
                    continue
                else:
                    denom = diff[idx + 1] - diff[idx]
                    if denom == 0.0:
                        phi_cross = 0.5 * (phi_vals[idx] + phi_vals[idx + 1])
                    else:
                        phi_cross = phi_vals[idx] - diff[idx] * (
                            phi_vals[idx + 1] - phi_vals[idx]
                        ) / denom

                if abs(rank[idx, level_a] - rank[idx, level_b]) == 1 or abs(
                    rank[idx + 1, level_a] - rank[idx + 1, level_b]
                ) == 1:
                    flux_idx = idx if gap[idx] <= gap[idx + 1] else idx + 1
                    add_record("crossing", phi_cross, 0.0, level_a, level_b, flux_idx)

            for idx in range(1, n_flux - 1):
                if not np.isfinite(gap[idx - 1 : idx + 2]).all():
                    continue
                if abs(rank[idx, level_a] - rank[idx, level_b]) != 1:
                    continue
                is_local_min = (
                    gap[idx] <= gap[idx - 1]
                    and gap[idx] <= gap[idx + 1]
                    and (gap[idx] < gap[idx - 1] or gap[idx] < gap[idx + 1])
                )
                if not is_local_min:
                    continue
                if np.sign(diff[idx - 1]) != np.sign(diff[idx + 1]):
                    continue
                add_record("avoided", phi_vals[idx], gap[idx], level_a, level_b, idx)

    records.sort(key=lambda rec: (rec["phi"], rec["level_a"], rec["level_b"], rec["kind"]))
    deduped = []
    phi_tol = 0.55 * np.min(np.diff(phi_vals))
    for rec in records:
        duplicate = False
        for prev in deduped:
            same_pair = rec["level_a"] == prev["level_a"] and rec["level_b"] == prev["level_b"]
            if same_pair and abs(rec["phi"] - prev["phi"]) <= phi_tol:
                if rec["gap_ghz"] < prev["gap_ghz"]:
                    prev.update(rec)
                duplicate = True
                break
        if not duplicate:
            deduped.append(rec)

    return deduped


def main():
    n_L = 11
    n_flux = 101
    evals_count = 12
    beta_vals = np.linspace(0.0, 1.0, n_L)[1:-1]
    n_L = len(beta_vals)
    L_c_vals = beta_vals * L_tot
    phi_vals = np.linspace(-0.5, 0.5, n_flux)

    alpha_grid = np.full((n_flux, n_L), np.nan)
    chi_grid = np.full((n_flux, n_L), np.nan)
    crossing_records = []
    column_results = []

    total = n_L
    done = 0
    overall_start = time.perf_counter()
    print(
        f"Computing alpha/chi heatmaps and tracked flux-sweep crossings on "
        f"{n_L} beta columns x {n_flux} flux points "
        f"(< {MAX_OVERLAY_GAP_GHZ * 1e3:.0f} MHz gap)..."
    )
    print(
        f"Tracked-level labels are anchored at the sweet spot and only written "
        f"for |Phi| <= {LABEL_VALID_PHI_MAX:.2f} Phi0."
    )
    print(
        "Heatmap alpha/chi values are computed from the labeled tracked-energy "
        "rows built after each flux sweep, with pure resonator |0,n> rows "
        "held fixed at their sweet-spot energies."
    )
    for i, L_c in enumerate(L_c_vals):
        beta = L_c / L_tot
        beta_start = time.perf_counter()
        column = sweep_beta_column(
            L_c,
            phi_vals,
            evals_count=evals_count,
            beta_index=i + 1,
            beta_total=total,
        )
        if column is None:
            column_results.append(None)
            done += 1
            elapsed = time.perf_counter() - overall_start
            avg_per_beta = elapsed / done
            eta = avg_per_beta * (total - done)
            print(
                f"  {done}/{total} beta columns (failed at beta={beta:.6f}) | "
                f"elapsed={format_duration(elapsed)} | eta={format_duration(eta)}"
            )
            continue

        column["beta"] = beta
        alpha_grid[:, i] = column["alpha"]
        chi_grid[:, i] = column["chi"]
        column_results.append(column)
        column_crossings = detect_tracked_crossings(
            beta, phi_vals, column["tracked_evals"], column["tracked_labels"]
        )
        crossing_records.extend(column_crossings)

        done += 1
        beta_elapsed = time.perf_counter() - beta_start
        elapsed = time.perf_counter() - overall_start
        avg_per_beta = elapsed / done
        eta = avg_per_beta * (total - done)
        print(
            f"  {done}/{total} beta columns | beta={beta:.6f} | "
            f"crossings={len(column_crossings)} | "
            f"beta_time={format_duration(beta_elapsed)} | "
            f"elapsed={format_duration(elapsed)} | eta={format_duration(eta)}"
        )

    beta_vals = L_c_vals / L_tot

    alpha_mhz = alpha_grid * 1e3
    chi_mhz = chi_grid * 1e3
    abs_chi_mhz = np.abs(chi_mhz)

    if np.all(np.isnan(alpha_mhz)):
        alpha_vmin, alpha_vmax = 0.0, 1.0
    else:
        alpha_vmin = float(np.nanmin(alpha_mhz))
        alpha_vmax = float(np.nanmax(alpha_mhz))

    chi_norm = None
    if np.all(np.isnan(chi_mhz)):
        chi_vmin, chi_vmax = -1.0, 1.0
    else:
        chi_abs_max = min(
            float(np.nanmax(np.abs(chi_mhz))),
            CHI_DISPLAY_ABS_MAX_GHZ * 1e3,
        )
        chi_vmin, chi_vmax = -chi_abs_max, chi_abs_max
        if chi_abs_max > 0.0:
            # Symmetric log scaling keeps chi's sign while compressing large magnitudes.
            chi_linthresh = max(0.5, chi_abs_max * 0.02)
            chi_norm = SymLogNorm(
                linthresh=chi_linthresh,
                linscale=1.0,
                vmin=chi_vmin,
                vmax=chi_vmax,
                base=10.0,
            )

    abs_chi_vmax = (
        min(float(np.nanmax(abs_chi_mhz)), CHI_DISPLAY_ABS_MAX_GHZ * 1e3)
        if not np.all(np.isnan(abs_chi_mhz))
        else 1.0
    )
    abs_chi_norm = None
    finite_abs_chi = abs_chi_mhz[np.isfinite(abs_chi_mhz)]
    positive_abs_chi = finite_abs_chi[finite_abs_chi > 0.0]
    if positive_abs_chi.size:
        abs_chi_vmin = max(float(np.nanmin(positive_abs_chi)), 1e-3)
        if abs_chi_vmax > abs_chi_vmin:
            abs_chi_norm = LogNorm(vmin=abs_chi_vmin, vmax=abs_chi_vmax)

    fig_heatmaps, (ax_alpha_only, ax_chi_only, ax_abs_chi_only) = plt.subplots(
        1, 3, figsize=(20, 5)
    )
    fig, (ax_alpha, ax_chi, ax_abs_chi) = plt.subplots(1, 3, figsize=(20, 5))

    im_alpha = ax_alpha.pcolormesh(
        beta_vals,
        phi_vals,
        alpha_mhz,
        shading="auto",
        cmap="viridis",
        vmin=alpha_vmin,
        vmax=alpha_vmax,
    )
    ax_alpha.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_alpha.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_alpha.set_title(r"Anharmonicity $\alpha$ (MHz)")
    fig.colorbar(im_alpha, ax=ax_alpha, label=r"$\alpha$ (MHz)")

    im_chi = ax_chi.pcolormesh(
        beta_vals,
        phi_vals,
        chi_mhz,
        shading="auto",
        cmap="plasma",
        norm=chi_norm,
        vmin=None if chi_norm is not None else chi_vmin,
        vmax=None if chi_norm is not None else chi_vmax,
    )
    ax_chi.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_chi.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_chi.set_title(r"Dispersive shift $\chi$ (MHz)")
    fig.colorbar(im_chi, ax=ax_chi, label=r"$\chi$ (MHz)")

    im_alpha_only = ax_alpha_only.pcolormesh(
        beta_vals,
        phi_vals,
        alpha_mhz,
        shading="auto",
        cmap="viridis",
        vmin=alpha_vmin,
        vmax=alpha_vmax,
    )
    ax_alpha_only.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_alpha_only.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_alpha_only.set_title(r"Anharmonicity $\alpha$ (MHz)")
    fig_heatmaps.colorbar(im_alpha_only, ax=ax_alpha_only, label=r"$\alpha$ (MHz)")

    im_chi_only = ax_chi_only.pcolormesh(
        beta_vals,
        phi_vals,
        chi_mhz,
        shading="auto",
        cmap="plasma",
        norm=chi_norm,
        vmin=None if chi_norm is not None else chi_vmin,
        vmax=None if chi_norm is not None else chi_vmax,
    )
    ax_chi_only.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_chi_only.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_chi_only.set_title(r"Dispersive shift $\chi$ (MHz)")
    fig_heatmaps.colorbar(im_chi_only, ax=ax_chi_only, label=r"$\chi$ (MHz)")

    im_abs_chi = ax_abs_chi.pcolormesh(
        beta_vals,
        phi_vals,
        abs_chi_mhz,
        shading="auto",
        cmap="magma",
        norm=abs_chi_norm,
        vmin=None if abs_chi_norm is not None else 0.0,
        vmax=None if abs_chi_norm is not None else abs_chi_vmax,
    )
    ax_abs_chi.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_abs_chi.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_abs_chi.set_title(r"Magnitude of dispersive shift $|\chi|$ (MHz)")
    fig.colorbar(im_abs_chi, ax=ax_abs_chi, label=r"$|\chi|$ (MHz)")
    ax_abs_chi.set_xscale("log")
    ax_abs_chi.set_yscale("symlog", linthresh=1e-2, linscale=1.0, base=10.0)

    im_abs_chi_only = ax_abs_chi_only.pcolormesh(
        beta_vals,
        phi_vals,
        abs_chi_mhz,
        shading="auto",
        cmap="magma",
        norm=abs_chi_norm,
        vmin=None if abs_chi_norm is not None else 0.0,
        vmax=None if abs_chi_norm is not None else abs_chi_vmax,
    )
    ax_abs_chi_only.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_abs_chi_only.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_abs_chi_only.set_title(r"Magnitude of dispersive shift $|\chi|$ (MHz)")
    fig_heatmaps.colorbar(im_abs_chi_only, ax=ax_abs_chi_only, label=r"$|\chi|$ (MHz)")
    ax_abs_chi_only.set_xscale("log")
    ax_abs_chi_only.set_yscale("symlog", linthresh=1e-2, linscale=1.0, base=10.0)

    if crossing_records:
        beta_cross = np.array([rec["beta"] for rec in crossing_records], dtype=float)
        phi_cross = np.array([rec["phi"] for rec in crossing_records], dtype=float)
        kinds = np.array([rec["kind"] for rec in crossing_records], dtype=object)

        crossing_mask = kinds == "crossing"
        avoided_mask = kinds == "avoided"

        if np.any(crossing_mask):
            ax_alpha.scatter(
                beta_cross[crossing_mask],
                phi_cross[crossing_mask],
                s=28,
                marker="o",
                facecolors="none",
                edgecolors="k",
                linewidths=0.8,
                zorder=6,
            )
            ax_chi.scatter(
                beta_cross[crossing_mask],
                phi_cross[crossing_mask],
                s=28,
                marker="o",
                facecolors="none",
                edgecolors="w",
                linewidths=0.8,
                zorder=6,
            )
            ax_abs_chi.scatter(
                beta_cross[crossing_mask],
                phi_cross[crossing_mask],
                s=28,
                marker="o",
                facecolors="none",
                edgecolors="w",
                linewidths=0.8,
                zorder=6,
            )

        if np.any(avoided_mask):
            ax_alpha.scatter(
                beta_cross[avoided_mask],
                phi_cross[avoided_mask],
                s=24,
                marker="x",
                c="k",
                linewidths=0.8,
                zorder=6,
            )
            ax_chi.scatter(
                beta_cross[avoided_mask],
                phi_cross[avoided_mask],
                s=24,
                marker="x",
                c="w",
                linewidths=0.8,
                zorder=6,
            )
            ax_abs_chi.scatter(
                beta_cross[avoided_mask],
                phi_cross[avoided_mask],
                s=24,
                marker="x",
                c="w",
                linewidths=0.8,
                zorder=6,
            )

    if crossing_records:
        print(
            f"\nTracked-state crossing overlay points "
            f"(< {MAX_OVERLAY_GAP_GHZ * 1e3:.0f} MHz gap): "
            f"{len(crossing_records)} total"
        )
        for rec in crossing_records:
            print(
                "  "
                f"beta={rec['beta']:.6f}, "
                f"phi={rec['phi']:.6f}, "
                f"{rec['kind']}, "
                f"levels=({rec['level_a']},{rec['level_b']}), "
                f"labels=({format_label(rec['label_a'])}, {format_label(rec['label_b'])}), "
                f"gap={rec['gap_ghz'] * 1e3:.3f} MHz"
            )
    else:
        print("\nTracked-state crossing overlay points: none found.")

    for ax in (ax_alpha, ax_chi, ax_abs_chi, ax_alpha_only, ax_chi_only, ax_abs_chi_only):
        ax.set_aspect("auto")
        ax.tick_params(axis="both", labelsize=10)

    fig_heatmaps.suptitle(r"$\alpha$, $\chi$, and $|\chi|$ heatmaps", fontsize=14)
    fig_heatmaps.tight_layout()
    fig.suptitle(r"$\alpha$, $\chi$, and $|\chi|$ vs $\beta$ and flux", fontsize=14)
    fig.tight_layout()
    sweep_fig = plot_beta_flux_sweeps(phi_vals, column_results)

    os.makedirs("plot_output", exist_ok=True)
    sweep_csv_path = "plot_output/imet_alpha_chi_vs_beta_flux.csv"
    crossings_csv_path = "plot_output/imet_alpha_chi_vs_beta_flux_crossings.csv"
    heatmaps_only_path = "plot_output/imet_alpha_chi_vs_beta_flux_heatmaps_only.png"
    beta_sweeps_path = "plot_output/imet_alpha_chi_vs_beta_flux_tracked_sweeps.png"
    write_sweep_csv(sweep_csv_path, phi_vals, column_results)
    write_crossings_csv(crossings_csv_path, crossing_records)
    out_path = "plot_output/imet_alpha_chi_vs_beta_flux.png"
    fig_heatmaps.savefig(heatmaps_only_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if sweep_fig is not None:
        sweep_fig.savefig(beta_sweeps_path, dpi=150, bbox_inches="tight")
    print(f"Saved {sweep_csv_path}")
    print(f"Saved {crossings_csv_path}")
    print(f"Saved {heatmaps_only_path}")
    print(f"Saved {out_path}")
    if sweep_fig is not None:
        print(f"Saved {beta_sweeps_path}")
    plt.show()

if __name__ == "__main__":
    main()

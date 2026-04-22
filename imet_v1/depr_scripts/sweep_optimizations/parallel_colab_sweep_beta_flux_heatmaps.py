"""
Parallel iMET beta/flux heatmap sweep script.

This version parallelizes across beta columns. Each worker builds its own
circuit for a single beta value, sweeps all flux points for that column, and
returns the extracted alpha/chi values plus tracked-state crossing overlays.

Colab setup (run once in a notebook cell before executing this script):
    !pip install -q scqubits scipy matplotlib numpy
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import scqubits as scq
from scipy.optimize import linear_sum_assignment


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._devnull = open(os.devnull, "w")
        sys.stdout = self._devnull
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        self._devnull.close()


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


def fit_effective_params_2mode(
    evals_rel, tol_nonmultiple=0.20, omega_r_known=None, omega_r_hint=None
):
    E = np.array(sorted(evals_rel))
    if E[0] != 0.0:
        E = E - E[0]

    if omega_r_known is not None:
        omega_r = float(omega_r_known)
    elif omega_r_hint is not None:
        omega_r, _ = _identify_omega_r_from_hint(E, float(omega_r_hint))
    else:
        omega_r = _peak_spacing_from_pairwise_diffs(
            E[: min(len(E), 16)], min_diff=1.0, bin_width=0.02
        )

    omega_q = None
    for Ek in E[1:]:
        resid, _ = _nearest_int_multiple_residual(Ek, omega_r)
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
    return (
        omega_q * nq
        + 0.5 * alpha_q * nq * (nq - 1)
        + omega_r * nr
        + chi_qr * nq * nr
    )


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


L_r_orig = 0.50e-9
C_r = 0.80e-12
L_c_orig = 0.16e-9
L_J1 = 30.0e-9
L_J2 = 30.0e-9
C_J1 = 40e-15
C_J2 = 40e-15

L_tot = L_r_orig + L_c_orig

PHI0 = 2.067833848e-15
E_CHARGE = 1.602176634e-19
H = 6.62607015e-34
MAX_OVERLAY_GAP_GHZ = 0.1


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


def fit_and_label_tracked_levels(evals_rel: np.ndarray, omega_r_hint: float):
    params = fit_effective_params_2mode(evals_rel, omega_r_hint=omega_r_hint)
    assignments = assign_labels_2mode(evals_rel, params, include_chi=True)

    unmatched = list(assignments)
    tracked_labels = []
    for energy in evals_rel:
        if not unmatched:
            tracked_labels.append(None)
            continue
        best_idx = min(
            range(len(unmatched)),
            key=lambda idx: abs(unmatched[idx]["E"] - energy),
        )
        best = unmatched.pop(best_idx)
        tracked_labels.append((best["nq"], best["nr"]))

    return params, tracked_labels


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {sec:02d}s"
    return f"{sec:d}s"


def format_label(label):
    if label is None:
        return "unlabeled"
    return f"|{label[0]},{label[1]}>"


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


def sweep_beta_column(L_c: float, phi_vals: np.ndarray, evals_count: int):
    circ, flux_attr, L_r = build_circuit(L_c)
    omega_r_hint = 1.0 / (2 * np.pi * np.sqrt(L_r * C_r)) / 1e9

    n_flux = len(phi_vals)
    raw_evals = np.full((n_flux, evals_count), np.nan)
    tracked_evals = np.full((n_flux, evals_count), np.nan)

    for idx, phi_ext in enumerate(phi_vals):
        with HiddenPrints():
            setattr(circ, flux_attr, float(phi_ext))
            evals, evecs = circ.eigensys(evals_count=evals_count)

        evals = np.asarray(evals, dtype=float)
        raw_evals[idx] = evals - evals[0]

        evecs = np.asarray(evecs)
        if evecs.ndim == 1:
            evecs = evecs[:, np.newaxis]
        if evecs.shape[0] < evecs.shape[1]:
            evecs = evecs.T

        if idx == 0:
            tracked_evals[idx] = raw_evals[idx]
            evecs_prev = evecs
            continue

        overlap = np.abs(evecs.conj().T @ evecs_prev) ** 2
        row_ind, col_ind = linear_sum_assignment(1.0 - overlap)
        perm = np.empty_like(row_ind)
        perm[col_ind] = row_ind
        tracked_evals[idx] = raw_evals[idx, perm]
        evecs_prev = evecs[:, perm]

    alpha_col = np.full(n_flux, np.nan)
    chi_col = np.full(n_flux, np.nan)
    tracked_labels = []

    for idx in range(n_flux):
        try:
            params, labels = fit_and_label_tracked_levels(
                tracked_evals[idx], omega_r_hint=omega_r_hint
            )
        except Exception:
            tracked_labels.append([None] * evals_count)
            continue

        alpha_col[idx] = params["alpha_q"]
        chi_col[idx] = params["chi_qr"]
        tracked_labels.append(labels)

    beta = L_c / L_tot
    crossings = detect_tracked_crossings(beta, phi_vals, tracked_evals, tracked_labels)
    return alpha_col, chi_col, crossings


def write_sweep_csv(path, L_c_vals, phi_vals, alpha_grid, chi_grid):
    beta_vals = L_c_vals / L_tot
    abs_chi_grid = np.abs(chi_grid)

    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "beta",
                "L_c_H",
                "L_c_nH",
                "phi_ext_over_phi0",
                "alpha_GHz",
                "alpha_MHz",
                "chi_GHz",
                "chi_MHz",
                "abs_chi_GHz",
                "abs_chi_MHz",
            ]
        )
        for j, phi in enumerate(phi_vals):
            for i, L_c in enumerate(L_c_vals):
                alpha_ghz = alpha_grid[j, i]
                chi_ghz = chi_grid[j, i]
                abs_chi_ghz = abs_chi_grid[j, i]
                writer.writerow(
                    [
                        beta_vals[i],
                        L_c,
                        L_c * 1e9,
                        phi,
                        alpha_ghz,
                        alpha_ghz * 1e3,
                        chi_ghz,
                        chi_ghz * 1e3,
                        abs_chi_ghz,
                        abs_chi_ghz * 1e3,
                    ]
                )


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
                "gap_MHz",
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
                    rec["gap_ghz"] * 1e3,
                ]
            )


def _worker(task):
    column_index, L_c, phi_vals, evals_count = task
    started = time.perf_counter()
    beta = L_c / L_tot
    try:
        alpha_col, chi_col, crossings = sweep_beta_column(L_c, phi_vals, evals_count)
        return {
            "column_index": column_index,
            "L_c": L_c,
            "beta": beta,
            "alpha": alpha_col,
            "chi": chi_col,
            "crossings": crossings,
            "elapsed_s": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        return {
            "column_index": column_index,
            "L_c": L_c,
            "beta": beta,
            "alpha": None,
            "chi": None,
            "crossings": [],
            "elapsed_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def choose_worker_count(requested: int | None, n_tasks: int) -> int:
    if requested is not None:
        return max(1, min(requested, n_tasks))
    cpu_total = os.cpu_count() or 1
    return max(1, min(n_tasks, cpu_total, 4))


def running_inside_ipython() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def choose_start_method(requested: str | None) -> str:
    if requested:
        return requested

    available = mp.get_all_start_methods()
    if os.name == "posix" and "fork" in available:
        return "fork"
    if "spawn" in available:
        return "spawn"
    return available[0]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Parallel beta/flux sweep for iMET alpha/chi heatmaps"
    )
    parser.add_argument("--n-L", type=int, default=21, help="Grid points along beta axis")
    parser.add_argument("--n-flux", type=int, default=21, help="Grid points along flux axis")
    parser.add_argument(
        "--evals-count",
        type=int,
        default=10,
        help="Number of lowest eigenpairs to compute per flux point",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count (default: conservative auto choice)",
    )
    parser.add_argument(
        "--start-method",
        choices=("spawn", "fork", "forkserver"),
        default=None,
        help="Multiprocessing start method (default: auto-select for environment)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip plt.show() at the end",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unrecognized notebook args: {unknown}")
    return args


def main():
    args = parse_args()

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    # Force non-interactive backend so spawned worker processes don't try to
    # open a display (which crashes them in Colab/headless environments).
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Prevent nested parallelism: each worker uses one thread for BLAS/OpenMP
    # so N workers don't fight over the same CPU cores and OOM.
    for _threadvar in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_threadvar, "1")

    import matplotlib.pyplot as plt

    n_L = args.n_L
    n_flux = args.n_flux
    evals_count = args.evals_count

    L_c_vals = np.linspace(0, L_tot, n_L)
    phi_vals = np.linspace(-0.5, 0.5, n_flux)
    L_c_vals = np.clip(L_c_vals, 1e-15, L_tot - 1e-15)

    alpha_grid = np.full((n_flux, n_L), np.nan)
    chi_grid = np.full((n_flux, n_L), np.nan)
    crossing_records = []

    tasks = [
        (column_index, float(L_c), phi_vals.copy(), evals_count)
        for column_index, L_c in enumerate(L_c_vals)
    ]
    total = len(tasks)
    workers = choose_worker_count(args.workers, total)
    start_method = choose_start_method(args.start_method)

    print(
        f"Computing alpha/chi heatmaps and tracked flux-sweep crossings on "
        f"{n_L} beta columns x {n_flux} flux points with {workers} workers "
        f"(start_method={start_method}) "
        f"(< {MAX_OVERLAY_GAP_GHZ * 1e3:.0f} MHz gap)..."
    )
    print("Preparing beta-column tasks...")

    overall_start = time.perf_counter()
    completed = 0

    # In Jupyter/Colab, stdlib multiprocessing deadlocks (fork inherits locked
    # kernel mutexes) or fails (spawn can't import __main__ = the kernel).
    # loky solves both: cloudpickle avoids __main__ import, pthread_atfork
    # handlers clean up locks after fork. It ships with joblib (pre-installed).
    _loky_executor = None
    if running_inside_ipython():
        try:
            from loky import ProcessPoolExecutor as _LokyPPE
            _loky_executor = _LokyPPE
        except ImportError:
            pass

    try:
        if _loky_executor is not None:
            print(f"Launching loky process pool with {workers} workers...", flush=True)
            executor_cm = _loky_executor(max_workers=workers)
        else:
            mp_context = mp.get_context(start_method)
            print(f"Launching process pool with {workers} workers...", flush=True)
            executor_cm = ProcessPoolExecutor(max_workers=workers, mp_context=mp_context)
        with executor_cm as executor:
            print(f"Submitting {total} beta-column jobs...", flush=True)
            futures = {executor.submit(_worker, task): task[0] for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                column_index = result["column_index"]
                beta = result["beta"]
                completed += 1

                if result["error"] is None:
                    alpha_grid[:, column_index] = result["alpha"]
                    chi_grid[:, column_index] = result["chi"]
                    crossing_records.extend(result["crossings"])
                    status = (
                        f"crossings={len(result['crossings'])}, "
                        f"beta_time={format_duration(result['elapsed_s'])}"
                    )
                else:
                    status = (
                        f"failed: {result['error']}, "
                        f"beta_time={format_duration(result['elapsed_s'])}"
                    )

                elapsed = time.perf_counter() - overall_start
                avg_per_column = elapsed / completed
                eta = avg_per_column * (total - completed)
                print(
                    f"  {completed}/{total} beta columns | beta={beta:.6f} | "
                    f"{status} | elapsed={format_duration(elapsed)} | "
                    f"eta={format_duration(eta)}"
                , flush=True)
    except BrokenProcessPool:
        if workers == 1:
            raise
        print(
            "Process pool failed in this notebook environment. "
            "Retrying serially with --workers 1 for stability."
        , flush=True)
        completed = 0
        overall_start = time.perf_counter()
        for task in tasks:
            column_index = task[0]
            beta = task[1] / L_tot
            # skip columns already filled by any futures that succeeded before the pool broke
            if not np.all(np.isnan(alpha_grid[:, column_index])):
                completed += 1
                continue
            print(
                f"  running serial beta column {column_index + 1}/{total} "
                f"(beta={beta:.6f})...",
                flush=True,
            )
            result = _worker(task)
            column_index = result["column_index"]
            beta = result["beta"]
            completed += 1

            if result["error"] is None:
                alpha_grid[:, column_index] = result["alpha"]
                chi_grid[:, column_index] = result["chi"]
                crossing_records.extend(result["crossings"])
                status = (
                    f"crossings={len(result['crossings'])}, "
                    f"beta_time={format_duration(result['elapsed_s'])}"
                )
            else:
                status = (
                    f"failed: {result['error']}, "
                    f"beta_time={format_duration(result['elapsed_s'])}"
                )

            elapsed = time.perf_counter() - overall_start
            avg_per_column = elapsed / completed
            eta = avg_per_column * (total - completed)
            print(
                f"  {completed}/{total} beta columns | beta={beta:.6f} | "
                f"{status} | elapsed={format_duration(elapsed)} | "
                f"eta={format_duration(eta)}"
            , flush=True)

    beta_vals = L_c_vals / L_tot
    alpha_mhz = alpha_grid * 1e3
    chi_mhz = chi_grid * 1e3
    abs_chi_mhz = np.abs(chi_mhz)

    if np.all(np.isnan(alpha_mhz)):
        alpha_vmin, alpha_vmax = 0.0, 1.0
    else:
        alpha_vmin = float(np.nanmin(alpha_mhz))
        alpha_vmax = float(np.nanmax(alpha_mhz))

    if np.all(np.isnan(chi_mhz)):
        chi_vmin, chi_vmax = -1.0, 1.0
    else:
        chi_abs_max = float(np.nanmax(np.abs(chi_mhz)))
        chi_vmin, chi_vmax = -chi_abs_max, chi_abs_max

    abs_chi_vmax = (
        float(np.nanmax(abs_chi_mhz)) if not np.all(np.isnan(abs_chi_mhz)) else 1.0
    )

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
    plt.colorbar(im_alpha, ax=ax_alpha, label=r"$\alpha$ (MHz)")

    im_chi = ax_chi.pcolormesh(
        beta_vals,
        phi_vals,
        chi_mhz,
        shading="auto",
        cmap="plasma",
        vmin=chi_vmin,
        vmax=chi_vmax,
    )
    ax_chi.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_chi.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_chi.set_title(r"Dispersive shift $\chi$ (MHz)")
    plt.colorbar(im_chi, ax=ax_chi, label=r"$\chi$ (MHz)")

    im_abs_chi = ax_abs_chi.pcolormesh(
        beta_vals,
        phi_vals,
        abs_chi_mhz,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=abs_chi_vmax,
    )
    ax_abs_chi.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_abs_chi.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_abs_chi.set_title(r"Magnitude of dispersive shift $|\chi|$ (MHz)")
    plt.colorbar(im_abs_chi, ax=ax_abs_chi, label=r"$|\chi|$ (MHz)")

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
    plt.colorbar(im_alpha_only, ax=ax_alpha_only, label=r"$\alpha$ (MHz)")

    im_chi_only = ax_chi_only.pcolormesh(
        beta_vals,
        phi_vals,
        chi_mhz,
        shading="auto",
        cmap="plasma",
        vmin=chi_vmin,
        vmax=chi_vmax,
    )
    ax_chi_only.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_chi_only.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_chi_only.set_title(r"Dispersive shift $\chi$ (MHz)")
    plt.colorbar(im_chi_only, ax=ax_chi_only, label=r"$\chi$ (MHz)")

    im_abs_chi_only = ax_abs_chi_only.pcolormesh(
        beta_vals,
        phi_vals,
        abs_chi_mhz,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=abs_chi_vmax,
    )
    ax_abs_chi_only.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_abs_chi_only.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_abs_chi_only.set_title(r"Magnitude of dispersive shift $|\chi|$ (MHz)")
    plt.colorbar(im_abs_chi_only, ax=ax_abs_chi_only, label=r"$|\chi|$ (MHz)")

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

    os.makedirs("plot_output", exist_ok=True)
    sweep_csv_path = "plot_output/imet_alpha_chi_vs_beta_flux_parallel.csv"
    crossings_csv_path = "plot_output/imet_alpha_chi_vs_beta_flux_parallel_crossings.csv"
    heatmaps_only_path = "plot_output/imet_alpha_chi_vs_beta_flux_parallel_heatmaps_only.png"
    out_path = "plot_output/imet_alpha_chi_vs_beta_flux_parallel.png"

    write_sweep_csv(sweep_csv_path, L_c_vals, phi_vals, alpha_grid, chi_grid)
    write_crossings_csv(crossings_csv_path, crossing_records)
    fig_heatmaps.savefig(heatmaps_only_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"Saved {sweep_csv_path}")
    print(f"Saved {crossings_csv_path}")
    print(f"Saved {heatmaps_only_path}")
    print(f"Saved {out_path}")
    print(
        f"Completed sweep in {format_duration(time.perf_counter() - overall_start)}",
        flush=True,
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

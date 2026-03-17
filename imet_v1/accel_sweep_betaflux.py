"""
iMET alpha & chi heatmap generator
───────────────────────────────────
Updates over original script:
  1. CPU multiprocessing via multiprocessing.Pool
  2. Uses circ.eigenvals() instead of circ.eigensys() (no eigenvectors needed)
  3. Optional GPU-accelerated diagonalization via CuPy or JAX (auto-fallback)
"""

import numpy as np
import matplotlib.pyplot as plt
import scqubits as scq
import warnings
import sys
import os
import time
import multiprocessing as mp
from functools import partial

# ──────────────────────────────────────────────────────────────────────────────
# GPU backend selection: try CuPy first, then JAX, then fall back to NumPy/SciPy
# ──────────────────────────────────────────────────────────────────────────────
GPU_BACKEND = "numpy"  # default fallback

try:
    import cupy as cp
    from cupy.linalg import eigvalsh as _cupy_eigvalsh
    # Quick sanity check that a CUDA device is actually reachable
    cp.cuda.runtime.getDeviceCount()
    GPU_BACKEND = "cupy"
except Exception:
    try:
        import jax
        import jax.numpy as jnp
        from jax import devices as _jax_devices
        # Check for an actual GPU device
        if any(d.platform == "gpu" for d in _jax_devices()):
            GPU_BACKEND = "jax"
        else:
            GPU_BACKEND = "jax_cpu"  # JAX present but CPU-only
    except Exception:
        pass

# [init] print is deferred to main() to avoid firing in every spawned worker


# ──────────────────────────────────────────────────────────────────────────────
# GPU-accelerated eigenvalue helpers
# ──────────────────────────────────────────────────────────────────────────────
def eigvals_gpu(matrix: np.ndarray, num_evals: int) -> np.ndarray:
    """
    Return the lowest `num_evals` eigenvalues of a Hermitian matrix,
    dispatching to the best available backend (CuPy → JAX → NumPy).
    """
    if GPU_BACKEND == "cupy":
        return _eigvals_cupy(matrix, num_evals)
    elif GPU_BACKEND in ("jax", "jax_cpu"):
        return _eigvals_jax(matrix, num_evals)
    else:
        return _eigvals_numpy(matrix, num_evals)


def _eigvals_cupy(matrix: np.ndarray, num_evals: int) -> np.ndarray:
    """CuPy path: transfer to GPU, diagonalise, transfer back."""
    mat_gpu = cp.asarray(matrix)
    # eigvalsh returns all eigenvalues in ascending order; slice to num_evals
    all_evals = cp.linalg.eigvalsh(mat_gpu)
    result = cp.asnumpy(all_evals[:num_evals])
    # Free GPU memory eagerly
    del mat_gpu, all_evals
    cp.get_default_memory_pool().free_all_blocks()
    return result


def _eigvals_jax(matrix: np.ndarray, num_evals: int) -> np.ndarray:
    """JAX path: JIT-compiled Hermitian eigvals on GPU (or CPU fallback)."""
    mat_jax = jnp.asarray(matrix)
    all_evals = jnp.linalg.eigvalsh(mat_jax)
    return np.asarray(all_evals[:num_evals])


def _eigvals_numpy(matrix: np.ndarray, num_evals: int) -> np.ndarray:
    """Pure NumPy/SciPy fallback using LAPACK subset driver."""
    from scipy.linalg import eigvalsh as sp_eigvalsh
    return sp_eigvalsh(matrix, subset_by_index=[0, num_evals - 1])


# ──────────────────────────────────────────────────────────────────────────────
# Silence helper
# ──────────────────────────────────────────────────────────────────────────────
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


with HiddenPrints():
    from identification import fit_effective_params_2mode, assign_labels_2mode, energy_of_label

# ──────────────────────────────────────────────────────────────────────────────
# Physical parameters
# ──────────────────────────────────────────────────────────────────────────────
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

EVALS_COUNT = 16  # how many lowest eigenvalues we need


def inductive_energy_ghz(L_H: float) -> float:
    EJ_J = (PHI0 / (2 * np.pi)) ** 2 / L_H
    return EJ_J / (H * 1e9)


def charging_energy_ghz(C_F: float) -> float:
    EC_J = E_CHARGE ** 2 / (2 * C_F)
    return EC_J / (H * 1e9)


E_J1 = inductive_energy_ghz(L_J1)
E_J2 = inductive_energy_ghz(L_J2)
E_C1 = charging_energy_ghz(C_J1)
E_C2 = charging_energy_ghz(C_J2)
E_C_r = charging_energy_ghz(C_r)


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation – called once per (L_c, phi_ext) grid point
# ──────────────────────────────────────────────────────────────────────────────
def eval_alpha_chi(L_c: float, phi_ext: float, use_gpu: bool = False):
    """
    Build the circuit, diagonalise, and extract alpha & chi.

    Parameters
    ----------
    L_c : float        Coupling inductance in Henries.
    phi_ext : float    External flux in units of Φ₀.
    use_gpu : bool     If True, build the Hamiltonian manually and use the
                       GPU eigensolver.  If False, use scqubits' built-in
                       eigenvals() on CPU (still benefits from multiprocessing).
    """
    # Clamp L_c first, then derive L_r from the clamped value
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

    try:
        with HiddenPrints():
            circ = scq.Circuit(iMET_yaml, from_file=False, ext_basis="harmonic")
            circ.cutoff_n_1 = 6
            circ.cutoff_ext_2 = 10
            circ.cutoff_ext_3 = 10

            _flux_syms = getattr(circ, "external_fluxes", None)
            if _flux_syms is not None and len(_flux_syms) > 0:
                _flux_attr = str(_flux_syms[0])
                setattr(circ, _flux_attr, float(phi_ext))
            else:
                return np.nan, np.nan

            # ── Diagonalisation strategy ──────────────────────────────────
            if use_gpu and GPU_BACKEND in ("cupy", "jax", "jax_cpu"):
                # Build the full Hamiltonian matrix on CPU, ship to GPU
                ham = circ.hamiltonian().toarray()  # dense numpy array
                evals = eigvals_gpu(ham, EVALS_COUNT)
            else:
                # CPU path: eigenvals only (no eigenvectors)
                evals = circ.eigenvals(evals_count=EVALS_COUNT)

        evals = np.asarray(evals, dtype=np.float64)
        evals_rel = evals - evals[0]

        # Use L_r (not L_tot) for the resonator frequency hint
        hint = 1.0 / (2 * np.pi * np.sqrt(L_r * C_r)) / 1e9
        params = fit_effective_params_2mode(evals_rel, omega_r_hint=hint)
        alpha_q = params["alpha_q"]
        chi_qr = params["chi_qr"]
        return alpha_q, chi_qr

    except Exception:
        return np.nan, np.nan


# ──────────────────────────────────────────────────────────────────────────────
# Worker wrapper for multiprocessing (must be a top-level function / picklable)
# ──────────────────────────────────────────────────────────────────────────────
def _worker(args):
    """Unpack args tuple and call eval_alpha_chi.  Used by Pool.map."""
    L_c, phi_ext, use_gpu = args
    return eval_alpha_chi(L_c, phi_ext, use_gpu=use_gpu)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute α and χ heatmaps for iMET circuit"
    )
    parser.add_argument(
        "--n-L", type=int, default=51, help="Grid points along L_c axis"
    )
    parser.add_argument(
        "--n-flux", type=int, default=51, help="Grid points along flux axis"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of multiprocessing workers (default: cpu_count). "
            "Applies in BOTH CPU-only and GPU modes — controls how many "
            "circuit builds + diagonalisations run in parallel."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help=(
            "Use GPU-accelerated diagonalisation (CuPy or JAX) for each "
            "worker.  Combines with --workers: e.g. --gpu --workers 8 "
            "runs 8 parallel workers that each offload eigvalsh to the GPU."
        ),
    )
    args = parser.parse_args()

    n_L = args.n_L
    n_flux = args.n_flux
    use_gpu = args.gpu and GPU_BACKEND in ("cupy", "jax", "jax_cpu")

    # --workers controls parallelism regardless of GPU mode.
    # Default: cpu_count for CPU-only, min(4, cpu_count) for GPU
    # (a sane default to avoid VRAM exhaustion, but user can override).
    if args.workers is not None:
        n_workers = args.workers
    elif use_gpu:
        n_workers = min(4, mp.cpu_count())
    else:
        n_workers = mp.cpu_count()

    diag_label = f"GPU ({GPU_BACKEND})" if use_gpu else "CPU"
    print(f"[init] Diagonalisation backend: {GPU_BACKEND}")
    print(f"[main] Diagonalisation: {diag_label}  |  Workers: {n_workers}")

    L_c_vals = np.linspace(0, L_tot, n_L)
    phi_vals = np.linspace(-0.5, 0.5, n_flux)
    L_c_vals = np.clip(L_c_vals, 1e-15, L_tot - 1e-15)

    # Build flat list of (L_c, phi, use_gpu) tasks
    tasks = [
        (L_c, phi, use_gpu)
        for phi in phi_vals
        for L_c in L_c_vals
    ]
    total = len(tasks)
    print(f"Computing alpha and chi on {n_L} x {n_flux} grid ({total} points)…")

    # ── Parallel execution ────────────────────────────────────────────────
    t0 = time.perf_counter()

    # Use 'spawn' to avoid CUDA fork issues
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap(_worker, tasks, chunksize=4), 1):
            results.append(result)
            if i % 50 == 0 or i == total:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(
                    f"  {i}/{total}  "
                    f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining, "
                    f"{rate:.1f} pts/s)"
                )

    elapsed_total = time.perf_counter() - t0
    print(f"[main] Done in {elapsed_total:.1f}s  ({total/elapsed_total:.1f} pts/s)")

    # ── Reshape results ───────────────────────────────────────────────────
    alpha_grid = np.full((n_flux, n_L), np.nan)
    chi_grid = np.full((n_flux, n_L), np.nan)

    for idx, (alpha, chi) in enumerate(results):
        j = idx // n_L  # flux index
        i = idx % n_L   # L_c index
        alpha_grid[j, i] = alpha
        chi_grid[j, i] = chi

    # ── Plotting ──────────────────────────────────────────────────────────
    beta_vals = L_c_vals / L_tot

    alpha_mhz = alpha_grid * 1e3
    chi_mhz = chi_grid * 1e3

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

    fig, (ax_alpha, ax_chi) = plt.subplots(1, 2, figsize=(14, 5))

    im_alpha = ax_alpha.pcolormesh(
        beta_vals, phi_vals, alpha_mhz,
        shading="auto", cmap="viridis",
        vmin=alpha_vmin, vmax=alpha_vmax,
    )
    ax_alpha.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_alpha.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_alpha.set_title(r"Anharmonicity $\alpha$ (MHz)")
    plt.colorbar(im_alpha, ax=ax_alpha, label=r"$\alpha$ (MHz)")

    im_chi = ax_chi.pcolormesh(
        beta_vals, phi_vals, chi_mhz,
        shading="auto", cmap="plasma",
        vmin=chi_vmin, vmax=chi_vmax,
    )
    ax_chi.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_chi.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_chi.set_title(r"Dispersive shift $\chi$ (MHz)")
    plt.colorbar(im_chi, ax=ax_chi, label=r"$\chi$ (MHz)")

    for ax in (ax_alpha, ax_chi):
        ax.set_aspect("auto")
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(r"$\alpha$ and $\chi$ vs $\beta$ and flux (iMET)", fontsize=14)
    fig.tight_layout()

    os.makedirs("plot_output", exist_ok=True)
    out_path = "plot_output/imet_alpha_chi_vs_beta_flux.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[main] Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
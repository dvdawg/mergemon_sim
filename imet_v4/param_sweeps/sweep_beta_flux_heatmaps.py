import numpy as np
import matplotlib.pyplot as plt
import scqubits as scq
import warnings
import sys
import os

from circuit_from_design import apply_recommended_cutoffs


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


with HiddenPrints():
    # Use the v2 identification module (3‑mode fitter) on the iMET
    # spectrum to extract alpha_q and chi_qr.
    from identification import fit_effective_params_3mode


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


def eval_alpha_chi(L_c: float, phi_ext: float):
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
            apply_recommended_cutoffs(circ, periodic_cutoff=6, extended_cutoff=10)

            _flux_syms = getattr(circ, "external_fluxes", None)
            if _flux_syms is not None and len(_flux_syms) > 0:
                _flux_attr = str(_flux_syms[0])
                setattr(circ, _flux_attr, float(phi_ext))
            else:
                return np.nan, np.nan

            evals, _ = circ.eigensys(evals_count=16)

        evals = np.asarray(evals)
        evals_rel = evals - evals[0]

        # Use L_r (not L_tot) for the resonator frequency hint
        omega_r_hint = 1.0 / (2 * np.pi * np.sqrt(L_r * C_r)) / 1e9

        # Fit effective 3‑mode parameters and extract alpha_q, chi_qr
        params = fit_effective_params_3mode(
            evals_rel,
            omega_r_hint=omega_r_hint,
            omega_q_hint=None,
            omega_a_hint=None,
        )
        alpha_q = params["alpha_q"]
        chi_qr = params["chi_qr"]
        return alpha_q, chi_qr
    except Exception:
        return np.nan, np.nan


def main():
    n_L = 51
    n_flux = 51
    L_c_vals = np.linspace(0, L_tot, n_L)
    phi_vals = np.linspace(-0.5, 0.5, n_flux)

    L_c_vals = np.clip(L_c_vals, 1e-15, L_tot - 1e-15)

    alpha_grid = np.full((n_flux, n_L), np.nan)
    chi_grid = np.full((n_flux, n_L), np.nan)

    total = n_L * n_flux
    done = 0
    print(f"Computing alpha and chi on {n_L} x {n_flux} grid ({total} points)…")
    for j, phi in enumerate(phi_vals):
        for i, L_c in enumerate(L_c_vals):
            alpha_grid[j, i], chi_grid[j, i] = eval_alpha_chi(L_c, phi)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total}")

    L_c_nH = L_c_vals * 1e9
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
    ax_alpha.set_title(r"Anharmonicity $\alpha_q$ (MHz)")
    plt.colorbar(im_alpha, ax=ax_alpha, label=r"$\alpha_q$ (MHz)")

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
    ax_chi.set_title(r"Dispersive shift $\chi_{qr}$ (MHz)")
    plt.colorbar(im_chi, ax=ax_chi, label=r"$\chi_{qr}$ (MHz)")

    for ax in (ax_alpha, ax_chi):
        ax.set_aspect("auto")
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(
        r"$\alpha_q$ and $\chi_{qr}$ vs $\beta$ and flux (iMET, v2 identification)",
        fontsize=14,
    )
    fig.tight_layout()

    os.makedirs("plot_output", exist_ok=True)
    out_path = "plot_output/imet_alpha_chi_vs_beta_flux_v2.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

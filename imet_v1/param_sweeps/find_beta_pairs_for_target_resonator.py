from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imet_v1.core.identification import (
    assign_labels_2mode,
    energy_of_label,
    fit_effective_params_2mode,
)


# Starting point from imet_v1/core/diag.py
L_R_BASE = 0.022e-9
L_C_BASE = 0.05e-9
C_R_BASE = 3.5e-12
L_J1 = 30.0e-9
L_J2 = 30.0e-9
C_J1 = 40e-15
C_J2 = 40e-15

TARGET_F_R_GHZ = 10.008
VERIFY_WITH_SCQUBITS = False

PHI0 = 2.067833848e-15
E_CHARGE = 1.602176634e-19
H = 6.62607015e-34


def inductive_energy_ghz(L_H: float) -> float:
    ej_j = (PHI0 / (2 * np.pi)) ** 2 / L_H
    return ej_j / (H * 1e9)


def charging_energy_ghz(C_F: float) -> float:
    ec_j = E_CHARGE**2 / (2 * C_F)
    return ec_j / (H * 1e9)


def total_lc_resonator_freq_ghz(L_r: float, L_c: float, C_r: float) -> float:
    return 1.0 / (2 * np.pi * np.sqrt((L_r + L_c) * C_r)) / 1e9


def required_c_for_target_freq(L_total: float, f_target_ghz: float) -> float:
    omega = 2 * np.pi * f_target_ghz * 1e9
    return 1.0 / (omega**2 * L_total)


def build_beta_pairs(L_total: float, c_r: float, l_r_values: np.ndarray):
    rows = []
    for l_r in l_r_values:
        l_c = L_total - l_r
        if l_c <= 0.0:
            continue
        beta = l_c / L_total
        f_r = total_lc_resonator_freq_ghz(l_r, l_c, c_r)
        rows.append(
            {
                "L_r_nH": 1e9 * l_r,
                "L_c_nH": 1e9 * l_c,
                "beta": beta,
                "f_r_total_ghz": f_r,
            }
        )
    return rows


def verify_point_with_spectrum(L_r: float, L_c: float, C_r: float):
    import scqubits as scq

    e_j1 = inductive_energy_ghz(L_J1)
    e_j2 = inductive_energy_ghz(L_J2)
    e_l_c = inductive_energy_ghz(L_c)
    e_l_r = inductive_energy_ghz(L_r)
    e_c1 = charging_energy_ghz(C_J1)
    e_c2 = charging_energy_ghz(C_J2)
    e_c_r = charging_energy_ghz(C_r)

    yaml_str = f"""# iMET
branches:
- ["JJ", 1,4, {e_j1:.6g}, {e_c1:.6g}]
- ["JJ", 1,2, {e_j2:.6g}, {e_c2:.6g}]
- ["L", 2,4, {e_l_c:.6g}]
- ["L", 2,3, {e_l_r:.6g}]
- ["C", 3,4, {e_c_r:.6g}]
"""

    circ = scq.Circuit(yaml_str, from_file=False, ext_basis="harmonic")
    circ.cutoff_n_1 = 6
    circ.cutoff_ext_2 = 10
    circ.cutoff_ext_3 = 10

    flux_syms = getattr(circ, "external_fluxes", None)
    if flux_syms is not None and len(flux_syms) > 0:
        setattr(circ, str(flux_syms[0]), 0.0)

    evals, _ = circ.eigensys(evals_count=12)
    evals_rel = np.asarray(evals) - evals[0]

    omega_r_hint = total_lc_resonator_freq_ghz(L_r, L_c, C_r)
    params = fit_effective_params_2mode(evals_rel, omega_r_hint=omega_r_hint)
    labels = assign_labels_2mode(evals_rel, params, include_chi=True)
    e00 = energy_of_label(labels, 0, 0)
    e01 = energy_of_label(labels, 0, 1)
    return {
        "omega_r_fit_ghz": params["omega_r"],
        "f_r_label_ghz": (e01 - e00) if e00 is not None and e01 is not None else np.nan,
        "omega_q_fit_ghz": params["omega_q"],
        "alpha_q_ghz": params["alpha_q"],
        "chi_qr_ghz": params["chi_qr"],
    }


def print_table(rows):
    print(
        f"{'L_r (nH)':>10} {'L_c (nH)':>10} {'beta':>10} {'f_r,total (GHz)':>18}"
    )
    print("-" * 52)
    for row in rows:
        print(
            f"{row['L_r_nH']:10.4f} "
            f"{row['L_c_nH']:10.4f} "
            f"{row['beta']:10.6f} "
            f"{row['f_r_total_ghz']:18.6f}"
        )


if __name__ == "__main__":
    l_total = L_R_BASE + L_C_BASE
    current_f_total = total_lc_resonator_freq_ghz(L_R_BASE, L_C_BASE, C_R_BASE)
    c_r_exact = required_c_for_target_freq(l_total, TARGET_F_R_GHZ)

    print("Base iMET v1 point")
    print(f"  L_r   = {L_R_BASE * 1e9:.4f} nH")
    print(f"  L_c   = {L_C_BASE * 1e9:.4f} nH")
    print(f"  L_tot = {l_total * 1e9:.4f} nH")
    print(f"  C_r   = {C_R_BASE * 1e12:.6f} pF")
    print(f"  f_r,total(L_r + L_c, C_r) = {current_f_total:.6f} GHz")
    print()
    print(f"To hit {TARGET_F_R_GHZ:.6f} GHz exactly at fixed L_tot:")
    print(f"  required C_r = {c_r_exact * 1e12:.6f} pF")
    print(
        f"  delta from current C_r = {(c_r_exact - C_R_BASE) * 1e15:.3f} fF"
    )
    print()
    print(
        "Small-L_r beta sweep candidates at fixed L_tot "
        f"= {l_total * 1e9:.4f} nH:"
    )

    l_r_candidates_nh = np.array([0.005, 0.010, 0.015, 0.020, 0.022, 0.025, 0.030])
    rows = build_beta_pairs(l_total, c_r_exact, l_r_candidates_nh * 1e-9)
    print_table(rows)

    if VERIFY_WITH_SCQUBITS:
        print()
        print("Spectral verification at phi_ext = 0")
        for row in rows:
            l_r = row["L_r_nH"] * 1e-9
            l_c = row["L_c_nH"] * 1e-9
            verified = verify_point_with_spectrum(l_r, l_c, c_r_exact)
            print(
                f"beta={row['beta']:.6f} "
                f"omega_r_fit={verified['omega_r_fit_ghz']:.6f} GHz "
                f"f_r_label={verified['f_r_label_ghz']:.6f} GHz "
                f"alpha={verified['alpha_q_ghz']:.6f} GHz "
                f"chi={verified['chi_qr_ghz']:.6f} GHz"
            )

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scqubits as scq
from scipy.optimize import linear_sum_assignment, minimize_scalar

L_r  = 0.60e-9
C_r  = 0.80e-12
L_c  = 0.15e-9
L_J1 = 30.0e-9
L_J2 = 30.0e-9
C_J1 = 40e-15
C_J2 = 40e-15

PHI0 = 2.067833848e-15
E_CHARGE = 1.602176634e-19
H_PLANCK = 6.62607015e-34


def inductive_energy_ghz(L):
    return ((PHI0 / (2 * np.pi))**2 / L) / (H_PLANCK * 1e9)

def charging_energy_ghz(C):
    return (E_CHARGE**2 / (2 * C)) / (H_PLANCK * 1e9)


E_J1_ghz  = inductive_energy_ghz(L_J1)
E_J2_ghz  = inductive_energy_ghz(L_J2)
E_L_c_ghz = inductive_energy_ghz(L_c)
E_L_r_ghz = inductive_energy_ghz(L_r)
E_C1_ghz  = charging_energy_ghz(C_J1)
E_C2_ghz  = charging_energy_ghz(C_J2)
E_C_r_ghz = charging_energy_ghz(C_r)

L_r_eff = L_r + L_c
OMEGA_R = 1.0 / np.sqrt(L_r_eff * C_r)
OMEGA_R_GHZ = OMEGA_R / (2 * np.pi * 1e9)

C_J_eff = C_J1 + C_J2
E_C_eff = charging_energy_ghz(C_J_eff)
d_asym = (E_J1_ghz - E_J2_ghz) / (E_J1_ghz + E_J2_ghz)

def E_J_squid(phi_ext):
    phi_ext = np.asarray(phi_ext, dtype=float)
    arg = np.pi * phi_ext
    cos_val = np.cos(arg)
    with np.errstate(divide="ignore", invalid="ignore"):
        tan_val = np.tan(arg)
        E_J = (E_J1_ghz + E_J2_ghz) * np.abs(cos_val) * np.sqrt(
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


NQ_MAX = 4
NR_MAX = 8


def predict_2mode(phi_ext, chi=0.0):
    E_J = float(E_J_squid(phi_ext))
    eq = transmon_energy_levels(E_J, E_C_eff, n_max=NQ_MAX)
    preds = {}
    for nq in range(NQ_MAX + 1):
        for nr in range(NR_MAX + 1):
            preds[(nq, nr)] = eq[nq] + nr * OMEGA_R_GHZ + chi * nq * nr
    return preds


def assign_labels(evals_rel, phi_ext, chi):
    preds = predict_2mode(phi_ext, chi)
    cands = sorted(preds.items(), key=lambda kv: kv[1])
    N = len(evals_rel)
    n_cands = min(len(cands), N)

    cost = np.full((N, n_cands), 1e9)
    for i, E_meas in enumerate(evals_rel):
        for j, ((nq, nr), E_pred) in enumerate(cands[:n_cands]):
            cost[i, j] = abs(E_meas - E_pred)

    row_ind, col_ind = linear_sum_assignment(cost)

    labels = {}
    for r, c in zip(row_ind, col_ind):
        (nq, nr), E_pred = cands[c]
        labels[r] = (nq, nr, E_pred)
    return labels


def fit_chi(evals_sweet, phi_sweet=0.0):
    E = np.asarray(evals_sweet, dtype=float)
    E = E - E[0]
    def residual(chi_val):
        preds = predict_2mode(phi_sweet, chi=chi_val)
        cands = sorted(preds.items(), key=lambda kv: kv[1])
        N = min(len(E), len(cands))
        cost = np.array(
            [[abs(E[i] - cands[j][1]) for j in range(N)] for i in range(N)]
        )
        r, c = linear_sum_assignment(cost)
        return float(cost[r, c].sum())

    result = minimize_scalar(residual, bounds=(-0.5, 0.5), method="bounded")
    return float(result.x)


iMET_yaml = f"""# iMET: asymmetric SQUID transmon coupled to LC resonator
branches:
- ["JJ", 1,4, {E_J1_ghz:.6g}, {E_C1_ghz:.6g}]
- ["JJ", 1,2, {E_J2_ghz:.6g}, {E_C2_ghz:.6g}]
- ["L",  2,4, {E_L_c_ghz:.6g}]
- ["L",  2,3, {E_L_r_ghz:.6g}]
- ["C",  3,4, {E_C_r_ghz:.6g}]
"""

print("Building circuit...")
circ = scq.Circuit(iMET_yaml, from_file=False, ext_basis="harmonic")
circ.cutoff_n_1 = 10
circ.cutoff_ext_2 = 14
circ.cutoff_ext_3 = 14

flux_syms = circ.external_fluxes
flux_attrs = [str(s) for s in flux_syms]
if not flux_attrs:
    sys.exit("No external flux variables found - check circuit topology.")
print(f"External flux variables: {flux_attrs}")

for a in flux_attrs:
    setattr(circ, a, 0.0)

ev_base, _ = circ.eigensys(evals_count=5)
E1_base = (ev_base - ev_base[0])[1]
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

N_LEVELS = 12
N_FLUX = 101
phi_vals = np.linspace(-0.5, 0.5, N_FLUX)
mid_idx = N_FLUX // 2

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
all_evals = np.zeros((N_FLUX, N_LEVELS))
all_evals[0] = raw_evals[0]
evecs_prev = all_evecs[0]

for i in range(1, N_FLUX):
    evals = raw_evals[i].copy()
    evecs = all_evecs[i]
    O = np.abs(evecs.conj().T @ evecs_prev) ** 2
    row_ind, col_ind = linear_sum_assignment(1 - O)
    perm = np.empty_like(row_ind)
    perm[col_ind] = row_ind
    all_evals[i] = evals[perm]
    evecs_prev = evecs[:, perm]

print("Tracking complete.")

print("\nFitting dispersive coupling chi at sweet spot...")
chi_fit = fit_chi(all_evals[mid_idx], phi_sweet=phi_vals[mid_idx])
print(f"chi / 2pi = {chi_fit * 1e3:.2f} MHz")

print("Assigning |nq, nr> labels at each flux point...")
level_labels_flux = []
for i, phi in enumerate(phi_vals):
    E_rel = all_evals[i] - all_evals[i, 0]
    linfo = assign_labels(E_rel, phi, chi_fit)
    d = {k: (nq, nr) for k, (nq, nr, _) in linfo.items()}
    level_labels_flux.append(d)

alpha_vals = np.full(N_FLUX, np.nan)
chi_vals = np.full(N_FLUX, np.nan)

for i in range(N_FLUX):
    labels = level_labels_flux[i]
    qn_to_energy = {}
    for k, (nq, nr) in labels.items():
        qn_to_energy[(nq, nr)] = all_evals[i, k]

    E_00 = qn_to_energy.get((0, 0))
    E_10 = qn_to_energy.get((1, 0))
    E_20 = qn_to_energy.get((2, 0))
    E_01 = qn_to_energy.get((0, 1))
    E_11 = qn_to_energy.get((1, 1))

    if E_00 is not None and E_10 is not None and E_20 is not None:
        alpha_vals[i] = (E_20 - E_10) - (E_10 - E_00)

    if E_00 is not None and E_10 is not None and E_01 is not None and E_11 is not None:
        chi_vals[i] = (E_11 - E_01) - (E_10 - E_00)

alpha_mhz = alpha_vals * 1e3
chi_mhz = chi_vals * 1e3

print(f"\n--- Exact values at sweet spot (Phi = {phi_vals[mid_idx]:.3f} Phi_0) ---")
if not np.isnan(alpha_vals[mid_idx]):
    print(f"  alpha / 2pi = {alpha_mhz[mid_idx]:.2f} MHz")
if not np.isnan(chi_vals[mid_idx]):
    print(f"  chi   / 2pi = {chi_mhz[mid_idx]:.2f} MHz")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(phi_vals, chi_mhz, color="C0", linewidth=1.6)
ax1.set_ylabel(r"$\chi\,/\,2\pi$ (MHz)", fontsize=12)
ax1.set_title("Dispersive Shift vs External Flux", fontsize=13)
ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax1.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax1.set_ylim(-5, 0)
ax1.grid(True, alpha=0.25)

ax2.plot(phi_vals, alpha_mhz, color="C1", linewidth=1.6)
ax2.set_xlabel(r"External flux  $\Phi_\mathrm{ext}\,/\,\Phi_0$", fontsize=12)
ax2.set_ylabel(r"$\alpha\,/\,2\pi$ (MHz)", fontsize=12)
ax2.set_title("Anharmonicity vs External Flux", fontsize=13)
ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax2.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
ax2.set_ylim(-500, -200)
ax2.grid(True, alpha=0.25)

fig.tight_layout()

os.makedirs("plot_output", exist_ok=True)
out_path = "plot_output/chi_alpha_vs_flux.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
plt.show()

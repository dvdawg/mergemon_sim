import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scqubits as scq
from scipy.optimize import linear_sum_assignment, minimize_scalar

AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 14
LEGEND_FONTSIZE = 10
SUPTITLE_FONTSIZE = 18

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

print(f"Resonator mode: L_eff = {L_r_eff*1e9:.3f} nH,  C_r = {C_r*1e12:.2f} pF")
print(f"ω_r / 2π = {OMEGA_R_GHZ:.4f} GHz")

C_J_eff = C_J1 + C_J2
E_C_eff = charging_energy_ghz(C_J_eff)
d_asym = (E_J1_ghz - E_J2_ghz) / (E_J1_ghz + E_J2_ghz)

print(f"\nQubit mode: C_eff = {C_J_eff*1e15:.1f} fF")
print(f"E_C = {E_C_eff:.4f} GHz")
print(f"E_J1 = {E_J1_ghz:.4f} GHz,  E_J2 = {E_J2_ghz:.4f} GHz")
print(f"asymmetry d = {d_asym:.4f}")


def E_J_squid(phi_ext: float | np.ndarray) -> np.ndarray:
    phi_ext = np.asarray(phi_ext, dtype=float)
    arg = np.pi * phi_ext
    cos_val = np.cos(arg)
    with np.errstate(divide="ignore", invalid="ignore"):
        tan_val = np.tan(arg)
        E_J = (E_J1_ghz + E_J2_ghz) * np.abs(cos_val) * np.sqrt(
            1.0 + d_asym**2 * tan_val**2
        )
    return np.where(np.isfinite(E_J), E_J, 0.0)


def transmon_energy_levels(E_J: float, E_C: float, n_max: int = 5) -> np.ndarray:
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
NR_MAX = 6 

def predict_2mode(phi_ext: float, chi: float = 0.0) -> dict:
    E_J = float(E_J_squid(phi_ext))
    eq  = transmon_energy_levels(E_J, E_C_eff, n_max=NQ_MAX) 
    preds = {}
    for nq in range(NQ_MAX + 1):
        for nr in range(NR_MAX + 1):
            preds[(nq, nr)] = eq[nq] + nr * OMEGA_R_GHZ + chi * nq * nr
    return preds

def assign_labels(evals_rel: np.ndarray, phi_ext: float, chi: float) -> dict:
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

def fit_chi(evals_sweet: np.ndarray, phi_sweet: float = 0.0) -> float:
    E = np.asarray(evals_sweet, dtype=float)
    E = E - E[0]
    def residual(chi_val):
        preds = predict_2mode(phi_sweet, chi=chi_val)
        cands = sorted(preds.items(), key=lambda kv: kv[1])
        N     = min(len(E), len(cands))
        cost  = np.array(
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

print("\nBuilding circuit…")
circ = scq.Circuit(iMET_yaml, from_file=False, ext_basis="harmonic")
circ.cutoff_n_1 = 6
circ.cutoff_ext_2 = 8
circ.cutoff_ext_3 = 8

flux_syms  = circ.external_fluxes
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
        print(f"{a} = 0.5 Φ₀  →  |Δf_01| = {shift:.4f} GHz")
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

print(f"\nSweeping {N_FLUX} flux points, {N_LEVELS} eigenvalues each…")
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

print("\nFitting dispersive coupling χ at sweet spot")
chi_fit = fit_chi(all_evals[mid_idx], phi_sweet=phi_vals[mid_idx])
print(f"χ / 2π = {chi_fit * 1e3:.2f} MHz")

print("Assigning |nq, nr⟩ labels at flux points")
level_labels_flux = []
for i, phi in enumerate(phi_vals):
    E_rel = all_evals[i] - all_evals[i, 0]
    linfo = assign_labels(E_rel, phi, chi_fit) 
    d = {k: (nq, nr) for k, (nq, nr, _) in linfo.items()}
    level_labels_flux.append(d)

print(f"\nState labels at Φ = {phi_vals[mid_idx]:.3f} Φ₀ (sweet spot):")
for k in range(1, N_LEVELS):
    nq, nr = level_labels_flux[mid_idx].get(k, (-1, -1))
    print(f"tracked level {k:2d} -> |{nq},{nr}⟩, E = {all_evals[mid_idx, k]:.4f} GHz")

all_qn_labels = set()
for d in level_labels_flux:
    all_qn_labels.update(d.values())

all_qn_labels.discard((0, 0))
all_qn_labels = sorted(all_qn_labels, key=lambda x: (x[0], x[1]))

n_colors = len(all_qn_labels)
cmap = plt.cm.tab20
colors = {lbl: cmap(i / max(n_colors - 1, 1)) for i, lbl in enumerate(all_qn_labels)}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
for lbl in all_qn_labels:
    nq, nr = lbl
    E_curve = np.full(N_FLUX, np.nan)
    for i in range(N_FLUX):
        for k in range(1, N_LEVELS):
            if level_labels_flux[i].get(k) == lbl:
                E_curve[i] = all_evals[i, k]
                break
    if not np.all(np.isnan(E_curve)):
        ax.plot(phi_vals, E_curve,
                color=colors[lbl], linewidth=1.4,
                label=f"|{nq},{nr}⟩")
    
ax.set_xlabel(r"External flux  $\Phi_\mathrm{ext}/\Phi_0$", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel("Energy (GHz)", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_title("Energy Levels vs External Flux", fontsize=TITLE_FONTSIZE)
ax.legend(frameon=False, ncols=2, fontsize=LEGEND_FONTSIZE, loc="upper right")
ax.grid(True, alpha=0.25)

ax2 = axes[1]
for lbl in all_qn_labels:
    nq, nr = lbl
    E_pred = np.array([
        predict_2mode(phi, chi=chi_fit).get((nq, nr), np.nan)
        for phi in phi_vals
    ])
    ax2.plot(phi_vals, E_pred,
             color=colors[lbl], linewidth=1.4, linestyle="--",
             label=f"|{nq},{nr}⟩")

ax2.set_xlabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_ylabel("Energy (GHz)", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_title(
    "Analytical two-mode model", fontsize=TITLE_FONTSIZE
)
ax2.legend(frameon=False, ncols=2, fontsize=LEGEND_FONTSIZE, loc="upper right")
ax2.grid(True, alpha=0.25)

ymax = max(ax.get_ylim()[1], ax2.get_ylim()[1])
for a in axes:
    a.set_ylim(-0.3, ymax)
    a.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

E_J_sweet = float(E_J_squid(0.0))
omega_q_sweet = float(transmon_energy_levels(E_J_sweet, E_C_eff, n_max=2)[1])
alpha_sweet = float(transmon_energy_levels(E_J_sweet, E_C_eff, n_max=2)[2] - 2 * transmon_energy_levels(E_J_sweet, E_C_eff, n_max=2)[1])

print("\n--- Analytical two-mode model parameters ---")
print(f"Resonator: ω_r/2π = {OMEGA_R_GHZ:.3f} GHz")
print(f"  L_eff = L_r + L_c = {L_r_eff*1e9:.2f} nH, C_r = {C_r*1e12:.1f} pF")
print(f"Qubit at sweet spot (Φ=0):")
print(f"  E_J^eff = {E_J_sweet:.3f} GHz")
print(f"  E_C = {E_C_eff:.4f} GHz")
print(f"  ω_q/2π = {omega_q_sweet:.3f} GHz")
print(f"  α_q = {alpha_sweet*1e3:.1f} MHz")
print(f"  χ = {chi_fit*1e3:.2f} MHz")

fig.suptitle(
    "Flux sweep plots", fontsize=SUPTITLE_FONTSIZE, y=0.93,
)
fig.tight_layout()

os.makedirs("plot_output", exist_ok=True)
out_path = "plot_output/imet_energy_levels_labelled.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
plt.show()

print(f"Resonator (SHO): ω_r/2π = {OMEGA_R_GHZ:.4f} GHz")
print(f"L_eff = L_r + L_c = {L_r_eff*1e9:.3f} nH")
print(f"C_r = {C_r*1e12:.3f} pF")
print(f"\n Qubit (SQUID transmon):")
print(f"C_eff = C_J1+C_J2 = {C_J_eff*1e15:.1f} fF")
print(f"E_C = {E_C_eff:.4f} GHz")
print(f"d (asymmetry) = {d_asym:.4f}")
print(f"E_J(Φ=0) = {E_J_sweet:.4f} GHz")
print(f"ω_q(Φ=0)/2π = {omega_q_sweet:.4f} GHz")
print(f"α_q(Φ=0) = {alpha_sweet*1e3:.1f} MHz")
print(f"\n Dispersive coupling (fit):")
print(f"χ/2π = {chi_fit*1e3:.2f} MHz")

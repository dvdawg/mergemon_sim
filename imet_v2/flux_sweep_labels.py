import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scqubits as scq
from scipy.optimize import linear_sum_assignment, minimize

from circuit_from_design import (
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
L_J_a, C_coupling_a = get_ancilla_params()

# Bare LC frequency hints (GHz):  omega = 1/sqrt(L*C)
OMEGA_R_GHZ = 1.0 / (np.sqrt(L_r   * C_r)           * 2 * np.pi * 1e9)
OMEGA_Q_GHZ = 1.0 / (np.sqrt(L_J_q * C_shunt_q)     * 2 * np.pi * 1e9)
OMEGA_A_GHZ = 1.0 / (np.sqrt(L_J_a * C_coupling_a)  * 2 * np.pi * 1e9)

print(f"Resonator: L_r={L_r*1e9:.2f} nH, C_r={C_r*1e12:.1f} pF  →  f_r_hint={OMEGA_R_GHZ:.4f} GHz")
print(f"Qubit:     L_J={L_J_q*1e9:.1f} nH, C={C_shunt_q*1e15:.0f} fF      →  f_q_hint={OMEGA_Q_GHZ:.4f} GHz")
print(f"Ancilla:   L_J={L_J_a*1e9:.1f} nH, C={C_coupling_a*1e15:.0f} fF     →  f_a_hint={OMEGA_A_GHZ:.4f} GHz")

E_J_data = inductive_energy_ghz(L_J_q)    # single SQUID junction E_J (GHz)
E_C_eff  = charging_energy_ghz(C_shunt_q) # qubit charging energy
d_asym   = 0.0

NQ_MAX = 4
NA_MAX = 3
NR_MAX = 6


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

    H = omega_q*nq + alpha_q/2*nq*(nq-1)
      + omega_a*na + alpha_a/2*na*(na-1)
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
            Ea = omega_a * na + 0.5 * alpha_a * na * (na - 1)
            for nr in range(NR_MAX + 1):
                E = (
                    eq[nq] + Ea + nr * OMEGA_R_GHZ
                    + chi_qa * nq * na
                    + chi_ar * na * nr
                    + chi_qr * nq * nr
                )
                preds[(nq, na, nr)] = E
    return preds


def assign_labels(evals_rel, phi_ext, analytical_params):
    preds   = predict_3mode(phi_ext, analytical_params)
    cands   = sorted(preds.items(), key=lambda kv: kv[1])
    N       = len(evals_rel)
    n_cands = min(len(cands), N)

    cost = np.full((N, n_cands), 1e9)
    for i, E_meas in enumerate(evals_rel):
        for j, ((nq, na, nr), E_pred) in enumerate(cands[:n_cands]):
            cost[i, j] = abs(E_meas - E_pred)

    row_ind, col_ind = linear_sum_assignment(cost)

    labels = {}
    for r, c in zip(row_ind, col_ind):
        (nq, na, nr), E_pred = cands[c]
        labels[r] = (nq, na, nr, E_pred)
    return labels


def fit_analytical_params_3mode(evals_sweet, phi_sweet=0.0, omega_a_hint=None):
    """
    Fit omega_a, alpha_a, chi_qa, chi_ar, chi_qr from the sweet-spot spectrum
    by minimising the Hungarian-assignment cost against the 3-mode prediction.

    omega_a_hint (GHz): physical LC estimate 1/(2*pi*sqrt(L_J_a * C_coupling)).
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
all_evals = np.zeros((N_FLUX, N_LEVELS))
all_evals[0] = raw_evals[0]
evecs_prev   = all_evecs[0]

for i in range(1, N_FLUX):
    evals = raw_evals[i].copy()
    evecs = all_evecs[i]
    O     = np.abs(evecs.conj().T @ evecs_prev) ** 2
    row_ind, col_ind = linear_sum_assignment(1 - O)
    perm = np.empty_like(row_ind)
    perm[col_ind] = row_ind
    all_evals[i]  = evals[perm]
    evecs_prev    = evecs[:, perm]

print("Tracking complete.")

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

# ── Assign |nq, na, nr> labels at each flux point ────────────────────────────
print("Assigning |nq, na, nr> labels at flux points")
level_labels_flux = []
for i, phi in enumerate(phi_vals):
    E_rel = all_evals[i] - all_evals[i, 0]
    linfo = assign_labels(E_rel, phi, analytical_params)
    d = {k: (nq, na, nr) for k, (nq, na, nr, _) in linfo.items()}
    level_labels_flux.append(d)

print(f"\nState labels at Phi = {phi_vals[mid_idx]:.3f} Phi_0 (sweet spot):")
for k in range(1, N_LEVELS):
    nq, na, nr = level_labels_flux[mid_idx].get(k, (-1, -1, -1))
    print(f"tracked level {k:2d} -> |{nq},{na},{nr}>, E = {all_evals[mid_idx, k]:.4f} GHz")

# ── Plotting ──────────────────────────────────────────────────────────────────
all_qn_labels = set()
for d in level_labels_flux:
    all_qn_labels.update(d.values())

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
            if level_labels_flux[i].get(k) == lbl:
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
        predict_3mode(phi, analytical_params).get((nq, na, nr), np.nan)
        for phi in phi_vals
    ])
    ax2.plot(
        phi_vals, E_pred,
        color=colors[lbl], linewidth=1.4, linestyle="--",
        label=f"|{nq},{na},{nr}>",
    )

ax2.set_xlabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_ylabel("Energy (GHz)", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_title("Analytical three-mode model", fontsize=TITLE_FONTSIZE)
ax2.legend(frameon=False, ncols=2, fontsize=LEGEND_FONTSIZE, loc="upper right")
ax2.grid(True, alpha=0.25)

ymax = max(ax.get_ylim()[1], ax2.get_ylim()[1])
for a in axes:
    a.set_ylim(-0.3, ymax)
    a.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

E_J_sweet     = float(E_J_squid(0.0))
omega_q_sweet = float(transmon_energy_levels(E_J_sweet, E_C_eff, n_max=2)[1])
alpha_q_sweet = float(
    transmon_energy_levels(E_J_sweet, E_C_eff, n_max=2)[2]
    - 2 * transmon_energy_levels(E_J_sweet, E_C_eff, n_max=2)[1]
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

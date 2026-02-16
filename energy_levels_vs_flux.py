import sys
import numpy as np
import matplotlib.pyplot as plt
import scqubits as scq
from scipy.optimize import linear_sum_assignment


L_r  = 0.60e-9
C_r  = 1.00e-12
L_c  = 0.15e-9
L_J1 = 18.3e-9
L_J2 = 11.0e-9
C_J1 = 55e-15
C_J2 = 33e-15

PHI0     = 2.067833848e-15
E_CHARGE = 1.602176634e-19
H_PLANCK = 6.62607015e-34


def inductive_energy_ghz(L):
    return ((PHI0 / (2 * np.pi))**2 / L) / (H_PLANCK * 1e9)


def charging_energy_ghz(C):
    return (E_CHARGE**2 / (2 * C)) / (H_PLANCK * 1e9)


E_J1  = inductive_energy_ghz(L_J1)
E_J2  = inductive_energy_ghz(L_J2)
E_L_c = inductive_energy_ghz(L_c)
E_L_r = inductive_energy_ghz(L_r)
E_C1  = charging_energy_ghz(C_J1)
E_C2  = charging_energy_ghz(C_J2)
E_C_r = charging_energy_ghz(C_r)


iMET_yaml = f"""# iMET: asymmetric SQUID transmon coupled to LC resonator
branches:
- ["JJ", 1,4, {E_J1:.6g}, {E_C1:.6g}]
- ["JJ", 1,2, {E_J2:.6g}, {E_C2:.6g}]
- ["L",  2,4, {E_L_c:.6g}]
- ["L",  2,3, {E_L_r:.6g}]
- ["C",  3,4, {E_C_r:.6g}]
"""


def _nearest_int_multiple_residual(E, omega):
    m = int(np.round(E / omega))
    return abs(E - m * omega), m


def _identify_omega_r_from_hint(E, hint):
    E = np.asarray(E)
    k = int(np.argmin(np.abs(E - hint)))
    return float(E[k]), k


def _peak_spacing_from_pairwise_diffs(energies, min_diff=1.0, bin_width=0.05):
    energies = np.array(sorted(energies))
    diffs = [float(energies[j] - energies[i])
             for i in range(len(energies))
             for j in range(i + 1, len(energies))
             if energies[j] - energies[i] >= min_diff]
    if not diffs:
        raise ValueError("Not enough diffs to estimate spacing.")
    diffs = np.array(diffs)
    nbins = max(10, int(np.ceil((diffs.max() - diffs.min()) / bin_width)))
    hist, edges = np.histogram(diffs, bins=nbins)
    k = int(np.argmax(hist))
    return 0.5 * (edges[k] + edges[k + 1])


def fit_effective_params_2mode(evals_rel, tol=0.20, omega_r_hint=None):
    E = np.array(sorted(evals_rel))
    E = E - E[0]
    if omega_r_hint is not None:
        omega_r, _ = _identify_omega_r_from_hint(E, float(omega_r_hint))
    else:
        omega_r = _peak_spacing_from_pairwise_diffs(
            E[:min(len(E), 16)], min_diff=1.0, bin_width=0.02)
    omega_q = None
    for Ek in E[1:]:
        resid, _ = _nearest_int_multiple_residual(Ek, omega_r)
        if resid > tol:
            omega_q = float(Ek)
            break
    if omega_q is None:
        omega_q = float(E[2]) if len(E) > 2 else float(E[1])
    idx_2q  = int(np.argmin(np.abs(E - 2.0 * omega_q)))
    alpha_q = float(E[idx_2q] - 2.0 * omega_q)
    idx_11  = int(np.argmin(np.abs(E - (omega_q + omega_r))))
    chi_qr  = float(E[idx_11] - (omega_q + omega_r))
    return dict(omega_r=omega_r, omega_q=omega_q, alpha_q=alpha_q, chi_qr=chi_qr)


def predicted_energy_2mode(nq, nr, p):
    return (p["omega_q"] * nq
            + 0.5 * p["alpha_q"] * nq * (nq - 1)
            + p["omega_r"] * nr
            + p["chi_qr"]  * nq * nr)


def assign_labels_2mode(evals_rel, p, nq_max=6, nr_max=10):
    # Use evals_rel in tracked order: do not sort. k is tracked index.
    E = np.asarray(evals_rel) - np.min(evals_rel)
    cands = sorted(
        [(predicted_energy_2mode(nq, nr, p), nq, nr)
         for nq in range(nq_max + 1)
         for nr in range(nr_max + 1)],
        key=lambda t: t[0],
    )
    used, out = set(), []
    for k in range(len(E)):
        Ek = E[k]
        best, best_score = None, 1e99
        for Ep, nq, nr in cands:
            if (nq, nr) in used:
                continue
            score = abs(Ep - Ek) + 1e-3 * (nq + nr)
            if score < best_score:
                best_score, best = score, (nq, nr, Ep)
        nq, nr, Ep = best
        used.add((nq, nr))
        out.append(dict(k=k, E=Ek, nq=nq, nr=nr, E_pred=Ep))
    return out


print("Building circuit…")
circ = scq.Circuit(iMET_yaml, from_file=False, ext_basis="harmonic")
circ.cutoff_n_1 = 6
circ.cutoff_ext_2 = 10
circ.cutoff_ext_3 = 10

flux_syms  = circ.external_fluxes
flux_attrs = [str(s) for s in flux_syms]

if not flux_attrs:
    sys.exit("No external flux variables found — check circuit topology.")

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
        print(f"  {a} = 0.5 Phi_0  ->  |Delta f01| = {shift:.4f} GHz")
        if shift > best_shift:
            best_shift, squid_attr = shift, a
    for a in flux_attrs:
        setattr(circ, a, 0.0)

print(f"\nSQUID loop flux variable identified: '{squid_attr}'")

for a in flux_attrs:
    if a != squid_attr:
        setattr(circ, a, 0.0)


N_LEVELS     = 12
OMEGA_R_HINT = 5.81

N_FLUX    = 101
phi_vals  = np.linspace(-0.5, 0.5, N_FLUX)
mid_idx   = N_FLUX // 2

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

print("Sweep complete.\n")

all_evals = np.zeros((N_FLUX, N_LEVELS))

print("Tracking states using eigenvector overlap…")

# Start from first flux point
evecs_prev = all_evecs[0]
all_evals[0] = raw_evals[0]

for i in range(1, N_FLUX):
    evals = raw_evals[i].copy()
    evecs = all_evecs[i]

    # Overlap matrix: O[j,k] = |<evec_j | evec_prev_k>|^2
    O = np.abs(evecs.conj().T @ evecs_prev) ** 2

    # Hungarian assignment to maximize total overlap
    cost = 1 - O
    row_ind, col_ind = linear_sum_assignment(cost)

    # Build permutation: reorder current states into previous-order
    perm = np.empty_like(row_ind)
    perm[col_ind] = row_ind

    # Apply permutation
    evals = evals[perm]
    evecs = evecs[:, perm]

    all_evals[i] = evals
    evecs_prev = evecs

print("Tracking complete.\n")

# Identify omega_r from sweet spot
omega_r, _ = _identify_omega_r_from_hint(all_evals[mid_idx], OMEGA_R_HINT)
print(f"Identified omega_r = {omega_r:.4f} GHz from sweet spot")

params_mid = fit_effective_params_2mode(all_evals[mid_idx], omega_r_hint=OMEGA_R_HINT)
params_mid["omega_r"] = omega_r

print(f"Effective parameters at sweet spot:")
print(f"  omega_r = {params_mid['omega_r']:.4f} GHz")
print(f"  omega_q = {params_mid['omega_q']:.4f} GHz")
print(f"  alpha_q = {params_mid['alpha_q']:.4f} GHz")
print(f"  chi_qr  = {params_mid['chi_qr']:.4f} GHz")

# Label states using 2-mode model
labels_info = assign_labels_2mode(all_evals[mid_idx], params_mid)
level_labels = {info["k"]: f"|{info['nq']},{info['nr']}⟩" for info in labels_info}

print(f"State labels at Phi = {phi_vals[mid_idx]:.3f} Phi_0 (sweet spot):")
for k in range(1, N_LEVELS):
    print(f"  tracked state {k:2d} -> {level_labels.get(k, f'|{k}>')}  "
          f"(E = {all_evals[mid_idx, k]:.4f} GHz)")


fig, ax = plt.subplots(figsize=(9, 5.5))

# Qualitative palette: 11 distinct colors (tab20 gives good separation)
colors = [plt.cm.tab20(i) for i in range(N_LEVELS - 1)]

for n in range(1, N_LEVELS):
    lbl = level_labels.get(n, f"|{n}⟩")
    ax.plot(phi_vals, all_evals[:, n],
            color=colors[n - 1], linewidth=1.3, label=lbl)

ax.set_xlabel(r"External flux  $\Phi_\mathrm{ext}/\Phi_0$", fontsize=13)
ax.set_ylabel("Energy  (GHz, relative to ground state)", fontsize=13)
ax.set_title("Energy levels vs external flux",
    fontsize=12,
)
ax.legend(frameon=False, ncols=2, fontsize=9, loc="upper right")
ax.grid(True, alpha=0.25)
fig.tight_layout()

plt.savefig("energy_levels_vs_flux.pdf", dpi=150)
plt.show()

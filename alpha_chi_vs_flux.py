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

N_FLUX    = 201
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
tracked_indices = np.zeros((N_FLUX, N_LEVELS), dtype=int)

all_evals[mid_idx] = raw_evals[mid_idx]
tracked_indices[mid_idx] = np.arange(N_LEVELS)

print("Tracking states using eigenvector overlap…")

prev_indices = np.arange(N_LEVELS)
for i in range(mid_idx + 1, N_FLUX):
    ev = raw_evals[i]
    evec = all_evecs[i]
    prev_evec = all_evecs[i - 1]

    O2 = np.abs(evec.conj().T @ prev_evec[:, prev_indices]) ** 2
    row_ind, col_ind = linear_sum_assignment(1 - O2)

    new_indices = np.zeros(N_LEVELS, dtype=int)
    new_indices[col_ind] = row_ind

    all_evals[i] = ev[new_indices]
    tracked_indices[i] = new_indices
    prev_indices = new_indices

prev_indices = np.arange(N_LEVELS)
for i in range(mid_idx - 1, -1, -1):
    ev = raw_evals[i]
    evec = all_evecs[i]
    prev_evec = all_evecs[i + 1]

    O2 = np.abs(evec.conj().T @ prev_evec[:, prev_indices]) ** 2
    row_ind, col_ind = linear_sum_assignment(1 - O2)

    new_indices = np.zeros(N_LEVELS, dtype=int)
    new_indices[col_ind] = row_ind

    all_evals[i] = ev[new_indices]
    tracked_indices[i] = new_indices
    prev_indices = new_indices

print("Tracking complete.\n")

# ── Label sweet-spot states using 2-mode model ────────────────────────────────
sweet_evecs = all_evecs[mid_idx]
omega_r, _ = _identify_omega_r_from_hint(all_evals[mid_idx], OMEGA_R_HINT)
params_mid = fit_effective_params_2mode(all_evals[mid_idx], omega_r_hint=OMEGA_R_HINT)
params_mid["omega_r"] = omega_r

print(f"Identified omega_r = {omega_r:.4f} GHz from sweet spot")
print(f"Effective parameters at sweet spot:")
print(f"  omega_r = {params_mid['omega_r']:.4f} GHz")
print(f"  omega_q = {params_mid['omega_q']:.4f} GHz")
print(f"  alpha_q = {params_mid['alpha_q']:.4f} GHz")
print(f"  chi_qr  = {params_mid['chi_qr']:.4f} GHz")

energy_std = np.std(all_evals, axis=0)
median_var = np.median(energy_std[1:])

flat_states = []
curved_states = []
for k in range(N_LEVELS):
    if energy_std[k] < median_var * 0.1:
        flat_states.append((k, all_evals[mid_idx, k]))
    else:
        curved_states.append((k, all_evals[mid_idx, k]))

flat_states.sort(key=lambda x: x[1])
curved_states.sort(key=lambda x: x[1])

label_to_idx = {}
for nr, (k, _E) in enumerate(flat_states):
    label_to_idx[(0, nr)] = k

all_cands = sorted(
    [(predicted_energy_2mode(nq, nr, params_mid), nq, nr)
     for nq in range(7) for nr in range(11)],
    key=lambda t: t[0],
)
used_labels = set((0, nr) for nr in range(len(flat_states)))

for k, Ek in curved_states:
    best, best_score = None, 1e99
    for Ep, nq, nr in all_cands:
        if (nq, nr) in used_labels:
            continue
        score = abs(Ep - Ek) + 1e-3 * (nq + nr)
        if score < best_score:
            best_score, best = score, (nq, nr)
    if best is not None:
        nq, nr = best
        used_labels.add((nq, nr))
        label_to_idx[(nq, nr)] = k

sweet_nq = np.zeros(N_LEVELS)
sweet_nr = np.zeros(N_LEVELS)
for (nq, nr), k in label_to_idx.items():
    sweet_nq[k] = nq
    sweet_nr[k] = nr

print("\nState labels at sweet spot:")
for (nq, nr), k in sorted(label_to_idx.items()):
    print(f"  |{nq},{nr}⟩  ->  tracked state {k:2d}  "
          f"(E = {all_evals[mid_idx, k]:.4f} GHz)")


# ── Per-flux-point labeling via projection onto sweet-spot basis ──────────────
# At each flux point, compute |⟨tracked_k(Φ)|sweet_j(0)⟩|² to get the
# bare-state content of each tracked eigenstate, then assign labels
# (nq, nr) = (round(⟨nq⟩), round(⟨nr⟩)) and extract alpha and chi.
target_labels = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
target_nq = np.array([t[0] for t in target_labels])
target_nr = np.array([t[1] for t in target_labels])
n_targets = len(target_labels)

alpha = np.zeros(N_FLUX)
chi = np.zeros(N_FLUX)
min_max_overlap = 1.0

for i in range(N_FLUX):
    tracked_evec = all_evecs[i][:, tracked_indices[i]]
    O2 = np.abs(tracked_evec.conj().T @ sweet_evecs) ** 2
    nq_exp = O2 @ sweet_nq
    nr_exp = O2 @ sweet_nr

    # Hungarian assignment: match tracked states to target labels
    cost = np.zeros((N_LEVELS, n_targets))
    for t in range(n_targets):
        cost[:, t] = (nq_exp - target_nq[t])**2 + (nr_exp - target_nr[t])**2

    row_ind, col_ind = linear_sum_assignment(cost)

    # Track overlap quality for the assigned states
    for m in range(len(row_ind)):
        min_max_overlap = min(min_max_overlap, O2[row_ind[m]].max())

    state_map = {}
    for m in range(len(col_ind)):
        state_map[target_labels[col_ind[m]]] = row_ind[m]

    E = all_evals[i]
    e00 = E[state_map[(0, 0)]]
    e10 = E[state_map[(1, 0)]]
    e20 = E[state_map[(2, 0)]]
    e01 = E[state_map[(0, 1)]]
    e11 = E[state_map[(1, 1)]]

    alpha[i] = (e20 - e10) - (e10 - e00)
    chi[i] = (e11 - e01) - (e10 - e00)

alpha_mhz = alpha * 1e3
chi_mhz   = chi * 1e3

print(f"\nTracking quality: min dominant overlap = {min_max_overlap:.4f}")
if min_max_overlap < 0.5:
    print("  WARNING: low overlap detected — consider increasing N_FLUX "
          "or N_LEVELS for better tracking near hybridization regions.")
print(f"\nAt sweet spot (Phi = 0):")
print(f"  alpha = {alpha_mhz[mid_idx]:.3f} MHz")
print(f"  chi   = {chi_mhz[mid_idx]:.3f} MHz")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

ax1.plot(phi_vals, alpha_mhz, color="tab:blue", linewidth=1.5)
ax1.set_ylabel(r"Anharmonicity $\alpha$  (MHz)", fontsize=13)
ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax1.grid(True, alpha=0.25)
ax1.set_title("Qubit anharmonicity and dispersive shift vs external flux",
              fontsize=12)

ax2.plot(phi_vals, chi_mhz, color="tab:red", linewidth=1.5)
ax2.set_xlabel(r"External flux  $\Phi_\mathrm{ext}/\Phi_0$", fontsize=13)
ax2.set_ylabel(r"Dispersive shift $\chi$  (MHz)", fontsize=13)
ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax2.grid(True, alpha=0.25)

fig.tight_layout()
plt.savefig("alpha_chi_vs_flux.pdf", dpi=150)
print("\nSaved: alpha_chi_vs_flux.pdf")
plt.show()

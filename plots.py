import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# ----------------------------
# Helpers: ladder operators
# ----------------------------
def destroy(N):
    a = np.zeros((N, N), dtype=complex)
    for n in range(1, N):
        a[n-1, n] = np.sqrt(n)
    return a

def kron3(A, B, C):
    return np.kron(np.kron(A, B), C)

def number_op(N):
    a = destroy(N)
    return a.conj().T @ a

# ----------------------------
# Your physical params (SI)
# ----------------------------
L_r  = 0.60e-9
C_r  = 1.00e-12
L_J1 = 18.3e-9
L_J2 = 11.0e-9
C_J1 = 55e-15
C_J2 = 33e-15

PHI0 = 2.067833848e-15
e    = 1.602176634e-19
h    = 6.62607015e-34

# Bare resonator frequency (Hz, GHz)
omega_r = 1.0 / np.sqrt(L_r * C_r)               # rad/s
f_r_ghz  = omega_r / (2*np.pi) / 1e9             # GHz

# SQUID: EJ from inductance at zero flux, then EJ_eff(Phi)
# Using SI energies (Joules) here.
EJ1_J = (PHI0/(2*np.pi))**2 / L_J1
EJ2_J = (PHI0/(2*np.pi))**2 / L_J2

C_c = C_J1 + C_J2
ECc_J = e**2/(2*C_c)

def EJ_eff_J(phi_over_phi0):
    """Asymmetric SQUID effective EJ (Joules) vs phi/phi0."""
    phi = phi_over_phi0 * PHI0
    return np.sqrt(EJ1_J**2 + EJ2_J**2 + 2*EJ1_J*EJ2_J*np.cos(2*np.pi*phi_over_phi0))

def LJeff(phi_over_phi0):
    """Effective Josephson inductance from EJ_eff."""
    return (PHI0/(2*np.pi))**2 / EJ_eff_J(phi_over_phi0)

def f_c_ghz(phi_over_phi0):
    """Coupler plasma frequency ~ 1/sqrt(LJ_eff * C_c) (GHz)."""
    omega_c = 1.0 / np.sqrt(LJeff(phi_over_phi0) * C_c)
    return omega_c/(2*np.pi)/1e9

# ----------------------------
# Effective model parameters (GHz)
# ----------------------------
# You can set these from your scqubits diagonalization at a chosen flux point.
# Example: use your previously printed numbers for the qubit:
wq_ghz = 6.247435          # qubit frequency (GHz), example from your table
aq_ghz = -0.35             # qubit anharmonicity (GHz) ~ -E_C/h (set/fit as needed)

# Coupler anharmonicity: transmon-like approx alpha_c ~ -E_Cc/h (in GHz)
alpha_c_ghz = -(ECc_J/h)/1e9

# Cross-Kerr between q and c (GHz). Set/fit as needed.
chi_qc_ghz = -0.002        # -2 MHz as a starter

# Coupling strength g_rc (GHz). 60 MHz => 0.060 GHz is 2g? careful:
# In many plots "2g" is the splitting, so g = (2g)/2.
two_g_MHz = 60.0
g_rc_ghz = (two_g_MHz/2) / 1000.0  # GHz

# ----------------------------
# Truncations
# ----------------------------
Nq, Nr, Nc = 4, 8, 8   # increase if you need more accuracy

aq = destroy(Nq)
ar = destroy(Nr)
ac = destroy(Nc)

nq = number_op(Nq)
nr = number_op(Nr)
nc = number_op(Nc)

Iq = np.eye(Nq)
Ir = np.eye(Nr)
Ic = np.eye(Nc)

# Kerr operators a^\dagger^2 a^2 = n(n-1)
kerr_q = (aq.conj().T @ aq.conj().T @ aq @ aq)
kerr_c = (ac.conj().T @ ac.conj().T @ ac @ ac)

# Position-like (a + a^\dagger)
xr = ar + ar.conj().T
xc = ac + ac.conj().T

# ----------------------------
# Build Hamiltonian (GHz units: H/h)
# ----------------------------
def H_eff_ghz(phi_over_phi0):
    wc = f_c_ghz(phi_over_phi0)

    # Local terms
    Hq = wq_ghz * nq + 0.5 * aq_ghz * kerr_q
    Hr = f_r_ghz * nr
    Hc = wc * nc + 0.5 * alpha_c_ghz * kerr_c

    # Cross-Kerr q-c will be added via tensor product below

    # Assemble with tensor products
    H = np.zeros((Nq*Nr*Nc, Nq*Nr*Nc), dtype=complex)
    H += kron3(Hq, Ir, Ic)
    H += kron3(Iq, Hr, Ic)
    H += kron3(Iq, Ir, Hc)

    # chi_qc * nq * nc
    H += chi_qc_ghz * kron3(nq, Ir, nc)

    # g_rc (ar+ar†)(ac+ac†)
    H += g_rc_ghz * kron3(Iq, xr, xc)

    return H

# ----------------------------
# Sweep flux and diagonalize
# ----------------------------
phis = np.linspace(-0.5, 0.5, 251)   # phi_ext/phi0
n_eigs = 14

evals = np.zeros((len(phis), n_eigs))
for i, phi in enumerate(phis):
    H = H_eff_ghz(phi)
    w, _ = eigh(H)
    w = np.real(w)
    w -= w[0]          # shift ground to 0
    evals[i, :] = w[:n_eigs]

# ----------------------------
# Plot 1: spectrum vs flux
# ----------------------------
plt.figure()
for k in range(n_eigs):
    plt.plot(phis, evals[:, k], label=f'level {k}')
plt.xlabel(r'$\Phi_{\rm ext}/\Phi_0$')
plt.ylabel('Energy (GHz, shifted)')
plt.title('Effective-model spectrum vs flux')
plt.legend(title='Eigenlevels', loc='best', fontsize='small')
plt.tight_layout()
plt.show()

# ----------------------------
# Plot 2: avoided-crossing zoom + estimate 2g
# We look for the minimum splitting between levels 1 and 2 (tweak if needed).
# ----------------------------
k1, k2 = 1, 2
splitting_ghz = evals[:, k2] - evals[:, k1]
imin = np.argmin(np.abs(splitting_ghz))
phi_star = phis[imin]
two_g_est_MHz = np.abs(splitting_ghz[imin]) * 1000.0  # GHz -> MHz

# zoom window
win = 20
i0 = max(0, imin - win)
i1 = min(len(phis), imin + win + 1)

plt.figure()
plt.plot(phis[i0:i1], evals[i0:i1, k1], label=f'level {k1}')
plt.plot(phis[i0:i1], evals[i0:i1, k2], label=f'level {k2}')
plt.axvline(phi_star, linestyle='--')
plt.title(f'Avoided crossing near {phi_star:.4f} : 2g ≈ {two_g_est_MHz:.1f} MHz')
plt.xlabel(r'$\Phi_{\rm ext}/\Phi_0$')
plt.ylabel('Energy (GHz, shifted)')
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Plot 3: "dispersive shift" χ(Φ) (MHz)
# A robust numeric definition: resonator frequency conditioned on coupler excitation.
#
# Define:
#   ωr(nc=0) ≈ E(0,1_r,0_c)-E(0,0,0)
#   ωr(nc=1) ≈ E(0,1_r,1_c)-E(0,0,1_c)
# Then χ_rc = ωr(nc=1) - ωr(nc=0)
#
# In dressed spectrum, we don't have perfect labels; in this simple model
# the low levels tend to be:
#   0: |0,0,0>
#   1: mostly |0,1,0> or |0,0,1> depending on flux
# So for a cleaner χ, you'd label eigenstates by overlap.
# We'll implement overlap-labeling with bare Fock states.
# ----------------------------
def basis_index(nq_, nr_, nc_):
    return (nq_*Nr + nr_)*Nc + nc_

# Bare target states in the computational basis
ket_000 = basis_index(0,0,0)
ket_010 = basis_index(0,1,0)  # 1 photon in resonator
ket_001 = basis_index(0,0,1)  # 1 excitation in coupler
ket_011 = basis_index(0,1,1)

chi_MHz = np.zeros(len(phis))

for i, phi in enumerate(phis):
    H = H_eff_ghz(phi)
    w, v = eigh(H)
    w = np.real(w)

    # Find the dressed eigenstate closest to each bare ket by maximum overlap
    overlaps = np.abs(v.conj().T)  # (dim, dim) but we use rows via v[:,j]
    # easier: compute overlap of each eigenvector with specific bare basis element
    ov_000 = np.abs(v[ket_000, :])**2
    ov_010 = np.abs(v[ket_010, :])**2
    ov_001 = np.abs(v[ket_001, :])**2
    ov_011 = np.abs(v[ket_011, :])**2

    j000 = np.argmax(ov_000)
    j010 = np.argmax(ov_010)
    j001 = np.argmax(ov_001)
    j011 = np.argmax(ov_011)

    # conditioned resonator frequencies (GHz)
    wr0 = (w[j010] - w[j000])  # resonator frequency when nc≈0
    wr1 = (w[j011] - w[j001])  # resonator frequency when nc≈1

    chi_MHz[i] = (wr1 - wr0) * 1000.0

plt.figure()
plt.plot(phis, chi_MHz)
plt.xlabel(r'$\Phi_{\rm ext}/\Phi_0$')
plt.ylabel(r'$\chi$ (MHz)')
plt.title('Dispersive-like shift vs flux (conditioned on coupler excitation)')
plt.legend([r'$\chi(\Phi)$'], loc='best')
plt.tight_layout()
plt.show()

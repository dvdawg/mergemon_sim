import numpy as np
import scqubits as scq

# params given
L_r  = 0.60e-9 
C_r  = 1.00e-12 
L_c  = 0.15e-9  
L_J1 = 18.3e-9  
L_J2 = 11.0e-9  
C_J1 = 55e-15
C_J2 = 33e-15   

# constants
PHI0 = 2.067833848e-15 
E_CHARGE = 1.602176634e-19 
H = 6.62607015e-34 

# conversions
def inductive_energy_ghz(L_H: float) -> float:
    """Return inductive energy EL / h in GHz for an inductor L (in Henries)."""
    EJ_J = (PHI0 / (2 * np.pi)) ** 2 / L_H  # Joules
    return EJ_J / (H * 1e9)                 # GHz


def charging_energy_ghz(C_F: float) -> float:
    """Return charging energy EC / h in GHz for a capacitor C (in Farads)."""
    EC_J = E_CHARGE ** 2 / (2 * C_F)  # Joules
    return EC_J / (H * 1e9)           # GHz


E_J1  = inductive_energy_ghz(L_J1)
E_J2  = inductive_energy_ghz(L_J2)
E_L_c = inductive_energy_ghz(L_c)
E_L_r = inductive_energy_ghz(L_r)
E_C1  = charging_energy_ghz(C_J1)
E_C2  = charging_energy_ghz(C_J2)
E_C_r = charging_energy_ghz(C_r)

print("Energy parameters (GHz):")
print(f"  E_J1  = {E_J1:.4f}")
print(f"  E_J2  = {E_J2:.4f}")
print(f"  E_C1  = {E_C1:.4f}")
print(f"  E_C2  = {E_C2:.4f}")
print(f"  E_L_c = {E_L_c:.4f}")
print(f"  E_L_r = {E_L_r:.4f}")
print(f"  E_C_r = {E_C_r:.4f}")

iMET_yaml = f"""# iMET
branches:
- ["JJ", 1,4, {E_J1:.6g}, {E_C1:.6g}]
- ["JJ", 1,2, {E_J2:.6g}, {E_C2:.6g}]
- ["L", 2,4, {E_L_c:.6g}]
- ["L", 2,3, {E_L_r:.6g}]
- ["C", 3,4, {E_C_r:.6g}]
"""
print(iMET_yaml)

def build_circuit(yaml_str: str):
    kwargs = dict(from_file=False, ext_basis="harmonic")
    return scq.Circuit(yaml_str, **kwargs)


circ = build_circuit(iMET_yaml)

try:
    circ.print_circuit()
except Exception:
    pass

try:
    cutoff_template = circ.get_cutoffs()
except Exception:
    cutoff_template = getattr(circ, "cutoffs", None) or getattr(circ, "truncation", None)

print("\Cutoff template:")
print(cutoff_template)

evals_count = 12
evals, evecs = circ.eigensys(evals_count=evals_count)
evals_rel = evals - evals[0]

print("\nLowest energies (GHz, shifted):")
for k, Ek in enumerate(evals_rel):
    print(f"  |{k:2d}>  {Ek:.6f} GHz")

f01 = evals_rel[1] - evals_rel[0]
f12 = evals_rel[2] - evals_rel[1]

anharmonicity = f12 - f01

omega_r = 1.0 / np.sqrt(L_r * C_r)
f_res_bare = omega_r / (2 * np.pi) / 1e9    # GHz

detuning = f01 - f_res_bare

print("\nTransitions & nonlinearities (GHz):")
print(f"  f01 (ground→1)         = {f01:.6f}")
print(f"  f12 (1→2)              = {f12:.6f}")
print(f"  anharmonicity α=f12-f01 = {anharmonicity:.6f}")
print("\nResonator & dispersive quantities (GHz):")
print(f"  bare resonator f_r     = {f_res_bare:.6f}")
print(f"  detuning Δ = f01 - f_r = {detuning:.6f}")
import numpy as np
import scqubits as scq
import warnings
import sys
import os

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

with HiddenPrints():
    from identification import fit_effective_params_2mode, assign_labels_2mode, energy_of_label

L_r_orig  = 0.50e-9
C_r  = 0.80e-12
L_c_orig  = 0.16e-9
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
    EC_J = E_CHARGE ** 2 / (2 * C_F) 
    return EC_J / (H * 1e9)

E_J1  = inductive_energy_ghz(L_J1)
E_J2  = inductive_energy_ghz(L_J2)
E_C1  = charging_energy_ghz(C_J1)
E_C2  = charging_energy_ghz(C_J2)
E_C_r = charging_energy_ghz(C_r)

def eval_for_Lc(L_c):
    L_r = L_tot - L_c
    
    if L_c < 1e-15: L_c = 1e-15
    if L_r < 1e-15: L_r = 1e-15

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
    
    with HiddenPrints():
        circ = scq.Circuit(iMET_yaml, from_file=False, ext_basis="harmonic")
        circ.cutoff_n_1 = 6
        circ.cutoff_ext_2 = 10
        circ.cutoff_ext_3 = 10
        
        _flux_syms = getattr(circ, "external_fluxes", None)
        if _flux_syms is not None and len(_flux_syms) > 0:
            _flux_attr = str(_flux_syms[0])
            setattr(circ, _flux_attr, 0.0)
        
        evals, _ = circ.eigensys(evals_count=16)
        
    evals = np.asarray(evals)
    evals_rel = evals - evals[0]
    
    try:
        # We can pass an approximate hint for the resonator based on the total LC resonance 
        # f_r_bare = 1.0 / (2 * np.pi * sqrt(L_tot * C_r)) / 1e9 (~6.9 GHz)
        hint = 1.0 / (2 * np.pi * np.sqrt(L_tot * C_r)) / 1e9
        
        params = fit_effective_params_2mode(evals_rel, omega_r_hint=hint)
        assign_chi = assign_labels_2mode(evals_rel, params, include_chi=True)
        
        E00 = energy_of_label(assign_chi, 0, 0)
        Eq10 = energy_of_label(assign_chi, 1, 0)
        Er01 = energy_of_label(assign_chi, 0, 1)
        
        f_q = (Eq10 - E00) if (Eq10 is not None and E00 is not None) else np.nan
        f_r = (Er01 - E00) if (Er01 is not None and E00 is not None) else np.nan
        alpha_q = params["alpha_q"]
        chi_qr = params["chi_qr"]
        
        return f_q, f_r, alpha_q, chi_qr
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan

if __name__ == "__main__":
    n_steps = 50   # 51 points: start (L_c=0, L_r=L_tot) then add/subtract delta each step
    delta = L_tot / n_steps
    print(f"Sweeping: start L_c=0, L_r={L_tot*1e9:.3f} nH; step by ±{delta*1e9:.4f} nH (L_tot = {L_tot*1e9:.3f} nH)")
    print(f"| {'L_c (nH)':^8} | {'L_r (nH)':^8} | {'f_q (GHz)':^10} | {'f_r (GHz)':^10} | {'alpha (GHz)':^12} | {'chi (GHz)':^10} |")
    print("-" * 78)
    for i in range(n_steps + 1):
        L_c = i * delta
        L_r = L_tot - L_c
        f_q, f_r, alpha_q, chi_qr = eval_for_Lc(L_c)
        print(f"| {L_c*1e9:8.4f} | {L_r*1e9:8.4f} | {f_q:10.6f} | {f_r:10.6f} | {alpha_q:12.6f} | {chi_qr:10.6f} |")
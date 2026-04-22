import numpy as np
import scqubits as scq

from circuit_from_design import (
    apply_recommended_cutoffs,
    build_circuit as build_circuit_from_design,
    get_resonator_params,
)

circ, iMET_yaml = build_circuit_from_design()
apply_recommended_cutoffs(circ, periodic_cutoff=12, extended_cutoff=18)
print(iMET_yaml)

L_r, C_r = get_resonator_params()

try :
    print ("Cutoff names:",getattr (circ ,"cutoff_names","Unknown"))
except Exception as e :
    print ("Error getting cutoff names:",e )

try :
    circ.print_circuit ()
except Exception :
    pass 

try :
    cutoff_template =circ .get_cutoffs ()
except Exception :
    cutoff_template =getattr (circ ,"cutoffs",None )or getattr (circ ,"truncation",None )

print("\nCutoff template:")
print (cutoff_template )


_flux_syms =getattr (circ ,"external_fluxes",None )
if _flux_syms is not None and len (_flux_syms )>0 :
    _flux_attr =str (_flux_syms [0 ])
    setattr (circ ,_flux_attr ,0.0 )
else :
    _flux_attr =None 

def get_spectrum (phi_ext ,evals_count =40 ):
    """Return (evals, evecs) in GHz for the circuit at external flux phi_ext (in units of Phi_0).
    evals: 1d array; evecs: (dim, evals_count) with columns = eigenstates.
    """
    if _flux_attr is not None :
        setattr (circ ,_flux_attr ,float (phi_ext ))
    evals ,evecs =circ .eigensys (evals_count =evals_count )
    evals =np .asarray (evals )
    evecs =np .asarray (evecs )
    if evecs .ndim ==1 :
        evecs =evecs [:,np .newaxis ]
    if evecs .shape [0 ]<evecs .shape [1 ]:
        evecs =evecs .T 
    return evals ,evecs 

evals_count =16 
evals ,evecs =circ .eigensys (evals_count =evals_count )
evals_rel =evals -evals [0 ]

print ("\nLowest energies (GHz, shifted):")
for k ,Ek in enumerate (evals_rel ):
    print (f"|{k :2d}> : {Ek :.6f} GHz")

f01 = evals_rel[1] - evals_rel[0]
f12 = evals_rel[2] - evals_rel[1]
# NOTE: for this 3-mode circuit the raw eigenvalue spacings f01 and f12 are
# not single-mode transitions; use identification.py for mode-resolved labeling.
anharmonicity = f12 - f01

omega_r    = 1.0 / np.sqrt(L_r * C_r)
f_res_bare = omega_r / (2 * np.pi) / 1e9

detuning = f01 - f_res_bare

f_r = omega_r / (2 * np.pi)
print(f"f_r = {f_r / 1e9:.6f} GHz")

print("\nRaw eigenvalue spacings (GHz) — see identification.py for mode labels:")
print(f"f01 (|0>→|1>)      = {f01:.6f}")
print(f"f12 (|1>→|2>)      = {f12:.6f}")
print(f"alpha_raw = f12-f01 = {anharmonicity:.6f}")
print("\nResonator & dispersive quantities (GHz):")
print(f"bare resonator f_res_bare = {f_res_bare:.6f}")
print(f"detuning Delta = f01 - f_res_bare = {detuning:.6f}")

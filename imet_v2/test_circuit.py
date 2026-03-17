import time
import numpy as np
import scqubits as scq

from circuit_from_design import build_circuit as build_circuit_from_design

circ, _ = build_circuit_from_design()
print("Cutoffs:", getattr(circ, "cutoff_names", []))
for name in getattr(circ, "cutoff_names", []):
    setattr(circ, name, 3)

start = time.time()
ev, _ = circ.eigensys(evals_count=3)
print(f"Eigensys complete in {time.time() - start:.2f}s.")

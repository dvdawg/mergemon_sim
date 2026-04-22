import time
import numpy as np
import scqubits as scq

from circuit_from_design import (
    apply_recommended_cutoffs,
    build_circuit as build_circuit_from_design,
)

circ, _ = build_circuit_from_design()
apply_recommended_cutoffs(circ, periodic_cutoff=8, extended_cutoff=12)
print("Cutoffs:", circ.get_cutoffs())

start = time.time()
ev, _ = circ.eigensys(evals_count=3)
print(f"Eigensys complete in {time.time() - start:.2f}s.")

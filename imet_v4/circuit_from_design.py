"""
Load scqubits circuit from imet_v2/design_graph.txt and build Circuit YAML.

design_graph.txt format: branches with physical units, e.g.:
  branches:
  - [JJ, 0, 1, 30nH]
  - [C, 0, 1, 45fF]
  - [L, 2, 3, 2.2nH]

JJ branches use inductance L_J (nH); E_J = (Phi0/(2*pi))**2/L_J. A default
junction capacitance (1 fF) is used for E_CJ unless overridden.
"""
import re
import os
import numpy as np
import scqubits as scq

PHI0 = 2.067833848e-15
E_CHARGE = 1.602176634e-19
H_PLANCK = 6.62607015e-34

# Default junction capacitance for JJ branches (Farads)
DEFAULT_C_J = 1.0e-15  # 1 fF

_UNIT_SCALE = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3,
    "k": 1e3, "M": 1e6, "G": 1e9,
}


def _parse_value(s: str):
    """Parse a value like '30nH' or '45fF' -> (float in SI, unit_str)."""
    s = s.strip()
    m = re.match(r"([\d.]+)\s*([fpnuµm]?H|[fpnuµm]?F)$", s, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse branch value: {s!r}")
    num, unit = float(m.group(1)), m.group(2).lower()
    if unit.endswith("h"):
        prefix = unit[0] if len(unit) > 1 else ""
        scale = _UNIT_SCALE.get(prefix, 1.0)
        return num * scale, "H"
    if unit.endswith("f"):
        prefix = unit[0] if len(unit) > 1 else ""
        scale = _UNIT_SCALE.get(prefix, 1.0)
        return num * scale, "F"
    raise ValueError(f"Unknown unit: {unit}")


def inductive_energy_ghz(L_H: float) -> float:
    """Inductive energy E_L or E_J in GHz for L in Henries."""
    E_J = (PHI0 / (2 * np.pi)) ** 2 / L_H
    return E_J / (H_PLANCK * 1e9)


def charging_energy_ghz(C_F: float) -> float:
    """Charging energy E_C in GHz for C in Farads."""
    E_C = E_CHARGE ** 2 / (2 * C_F)
    return E_C / (H_PLANCK * 1e9)


def load_design_graph(path: str = None) -> str:
    """
    Load design_graph.txt and return a scqubits YAML string (energies in GHz).
    path: path to design file; default is imet_v2/design_graph.txt next to this module.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "design_graph.txt")
    with open(path, "r") as f:
        content = f.read()

    # Parse YAML-like branches: "- [TYPE, n1, n2, value]" or "- [JJ, n1, n2, value]" (JJ may need ECJ)
    branch_lines = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Match "- [TYPE, 0, 1, 30nH]" or "- [JJ, 0, 1, 30nH]"
        m = re.match(r"-\s*\[\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.\w]+)\s*\]", line)
        if not m:
            continue
        btype, n1, n2, val = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        branch_lines.append((btype.upper(), n1, n2, val))

    yaml_branches = []
    for btype, n1, n2, val in branch_lines:
        value_si, unit = _parse_value(val)
        if btype == "C":
            E_C = charging_energy_ghz(value_si)
            yaml_branches.append(f'- ["C", {n1}, {n2}, {E_C:.6g}]')
        elif btype == "L":
            E_L = inductive_energy_ghz(value_si)
            yaml_branches.append(f'- ["L", {n1}, {n2}, {E_L:.6g}]')
        elif btype == "JJ":
            E_J = inductive_energy_ghz(value_si)  # L_J -> E_J
            E_CJ = charging_energy_ghz(DEFAULT_C_J)
            yaml_branches.append(f'- ["JJ", {n1}, {n2}, {E_J:.6g}, {E_CJ:.6g}]')
        else:
            raise ValueError(f"Unknown branch type: {btype}")

    yaml_str = "branches:\n" + "\n".join(yaml_branches) + "\n"
    return yaml_str


def build_circuit(from_file_path: str = None, ext_basis: str = "harmonic", **kwargs):
    """
    Build scqubits Circuit from design_graph.txt.
    Returns (circ, yaml_str).
    """
    yaml_str = load_design_graph(from_file_path)
    opts = dict(from_file=False, ext_basis=ext_basis, **kwargs)
    circ = scq.Circuit(yaml_str, **opts)
    return circ, yaml_str


def _parse_branches(path=None):
    """
    Parse all branches from design_graph.txt.
    Returns list of (btype, n1, n2, value_SI) where btype is 'JJ', 'C', or 'L'.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "design_graph.txt")
    with open(path, "r") as f:
        content = f.read()
    branches = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"-\s*\[\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.\w]+)\s*\]", line)
        if not m:
            continue
        btype = m.group(1).upper()
        n1, n2 = int(m.group(2)), int(m.group(3))
        val_si, _ = _parse_value(m.group(4))
        branches.append((btype, n1, n2, val_si))
    return branches


def get_qubit_params(path: str = None) -> tuple:
    """
    Return (L_J, C_shunt) in SI for the qubit SQUID arm.

    The qubit JJ is identified as the direct-shunted JJ with the smallest
    shunt capacitance (e.g. JJ(0,1) paired with C(0,1)). Returns the first
    such (L_J, C_shunt) pair found — for a symmetric SQUID all arms are equal.

    Bare qubit frequency hint:  omega_q = 1 / sqrt(L_J * C_shunt)
    """
    branches = _parse_branches(path)
    # Group by canonical node pair
    by_pair = {}
    for btype, n1, n2, val in branches:
        key = (min(n1, n2), max(n1, n2))
        by_pair.setdefault(key, []).append((btype, val))

    shunted_jjs = []
    for btype, n1, n2, val in branches:
        if btype != "JJ":
            continue
        key = (min(n1, n2), max(n1, n2))
        caps = [v for bt, v in by_pair.get(key, []) if bt == "C"]
        if caps:
            shunted_jjs.append((val, min(caps)))

    if shunted_jjs:
        return min(shunted_jjs, key=lambda item: item[1])

    raise ValueError("Could not find qubit JJ with direct shunt capacitor in design_graph")


def get_ancilla_params(path: str = None) -> tuple:
    """
    Return (L_J, C_ancilla) in SI for the ancilla mode.

    Prefer the ancilla JJ's direct shunt capacitor on the same node pair. If
    the design has no direct-shunted ancilla JJ, fall back to the legacy rule:
    choose the unshunted JJ and use the smallest adjacent capacitor that shares
    exactly one node with that JJ.

    Bare ancilla frequency hint:  omega_a = 1 / sqrt(L_J * C_ancilla)
    """
    branches = _parse_branches(path)
    by_pair = {}
    for btype, n1, n2, val in branches:
        key = (min(n1, n2), max(n1, n2))
        by_pair.setdefault(key, []).append((btype, val))

    shunted_jjs = []
    for btype, n1, n2, val in branches:
        if btype != "JJ":
            continue
        key = (min(n1, n2), max(n1, n2))
        direct_caps = [v for bt, v in by_pair.get(key, []) if bt == "C"]
        if direct_caps:
            shunted_jjs.append((val, min(direct_caps)))

    if len(shunted_jjs) >= 2:
        return max(shunted_jjs, key=lambda item: item[1])

    for btype, n1, n2, val in branches:
        if btype != "JJ":
            continue
        key = (min(n1, n2), max(n1, n2))
        direct_caps = [v for bt, v in by_pair.get(key, []) if bt == "C"]
        if direct_caps:
            continue

        adj_caps = []
        for bt2, n1_2, n2_2, val2 in branches:
            if bt2 != "C":
                continue
            key2 = (min(n1_2, n2_2), max(n1_2, n2_2))
            if key2 == key:
                continue
            if n1_2 == n1 or n1_2 == n2 or n2_2 == n1 or n2_2 == n2:
                adj_caps.append(val2)
        if adj_caps:
            return val, min(adj_caps)

    raise ValueError("Could not determine ancilla JJ and capacitance from design_graph")


def get_resonator_params(path: str = None) -> tuple:
    """
    Parse design_graph and return (L_r, C_r) in SI for the resonator.
    Uses the first L and the largest C in the design as the main resonator L and C
    (for a rough f_res_bare = 1/(2*pi*sqrt(L*C))).
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "design_graph.txt")
    with open(path, "r") as f:
        content = f.read()
    L_r, C_r = None, None
    for line in content.splitlines():
        m = re.match(r"-\s*\[\s*L\s*,\s*\d+\s*,\s*\d+\s*,\s*([\d.\w]+)\s*\]", line, re.IGNORECASE)
        if m:
            val, _ = _parse_value(m.group(1))
            L_r = val
            break
    for line in content.splitlines():
        m = re.match(r"-\s*\[\s*C\s*,\s*\d+\s*,\s*\d+\s*,\s*([\d.\w]+)\s*\]", line, re.IGNORECASE)
        if m:
            val, _ = _parse_value(m.group(1))
            if C_r is None or val > C_r:
                C_r = val
    if L_r is None or C_r is None:
        raise ValueError("Could not find L and C in design_graph for resonator")
    return L_r, C_r

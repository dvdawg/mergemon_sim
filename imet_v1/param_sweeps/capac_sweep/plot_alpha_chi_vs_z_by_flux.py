import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS = {
    "6.9 GHz": SCRIPT_DIR / "6.9ghz",
    "10 GHz": SCRIPT_DIR / "10ghz",
}
OUTPUT_DIR = SCRIPT_DIR / "plot_output"
OUTPUT_DIR.mkdir(exist_ok=True)

FLUX_BIASES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
CSV_BASENAME = "imet_alpha_chi_vs_z_flux"
SHADE_MAPS = {
    "6.9 GHz": plt.cm.Blues,
    "10 GHz": plt.cm.Reds,
}
METRICS = [
    ("alpha_mhz", r"Anharmonicity $\alpha$ (MHz)", "alpha"),
    ("chi_mhz", r"Dispersive shift $\chi$ (MHz)", "chi"),
    ("abs_chi_mhz", r"Magnitude of dispersive shift $|\chi|$ (MHz)", "abs_chi"),
]


def find_main_csv(folder: Path) -> Path:
    matches = sorted(
        path
        for path in folder.glob("*.csv")
        if "crossings" not in path.name and CSV_BASENAME in path.name
    )
    if not matches:
        raise FileNotFoundError(f"No sweep CSV found in {folder}")
    return matches[0]


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "frequency_ghz": float(row["frequency_GHz"]),
                    "capacitance_pf": float(row["capacitance_pF"]),
                    "l_tot_nh": float(row["L_tot_nH"]),
                    "z_ohm": float(row["Z_ohm"]),
                    "phi": float(row["phi_ext_over_phi0"]),
                    "alpha_mhz": float(row["alpha_GHz"]) * 1e3,
                    "chi_mhz": float(row["chi_GHz"]) * 1e3,
                    "abs_chi_mhz": float(row["abs_chi_GHz"]) * 1e3,
                }
            )
    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")
    return rows


def unique_sorted(values):
    return np.array(sorted(set(values)), dtype=float)


def nearest_value(values: np.ndarray, target: float):
    idx = int(np.argmin(np.abs(values - target)))
    return float(values[idx]), idx


def select_flux_slice(rows, target_flux: float):
    phi_values = unique_sorted(row["phi"] for row in rows)
    phi_used, _ = nearest_value(phi_values, target_flux)
    slice_rows = [row for row in rows if np.isclose(row["phi"], phi_used)]
    if not slice_rows:
        raise ValueError(f"No rows found near flux {target_flux}")
    slice_rows.sort(key=lambda row: row["z_ohm"])
    return phi_used, slice_rows


def load_dataset(dataset_label: str, folder: Path):
    csv_path = find_main_csv(folder)
    rows = load_rows(csv_path)
    return {
        "label": dataset_label,
        "folder": folder,
        "csv_path": csv_path,
        "rows": rows,
    }


def plot_metric(datasets, metric_key: str, y_label: str, output_stem: str):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    flux_count = len(FLUX_BIASES)

    for dataset_label, dataset in datasets.items():
        cmap = SHADE_MAPS[dataset_label]
        for flux_index, flux_target in enumerate(FLUX_BIASES):
            phi_used, flux_rows = select_flux_slice(dataset["rows"], flux_target)
            x_vals = np.array([row["z_ohm"] for row in flux_rows], dtype=float)
            y_vals = np.array([row[metric_key] for row in flux_rows], dtype=float)
            color = cmap(0.35 + 0.55 * flux_index / max(flux_count - 1, 1))
            ax.plot(
                x_vals,
                y_vals,
                marker="o",
                linewidth=1.8,
                markersize=5.5,
                color=color,
                label=f"{dataset_label}, $\\Phi_{{ext}}/\\Phi_0={phi_used:+.1f}$",
            )

    ax.set_xlabel(r"$Z=\sqrt{L_\mathrm{tot}/C}$ ($\Omega$)")
    ax.set_ylabel(y_label)
    ax.set_title(
        y_label.replace(" (MHz)", "") + r" vs $Z$ for selected flux biases"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncols=2, fontsize=9)
    fig.tight_layout()

    png_path = OUTPUT_DIR / f"{output_stem}_vs_z_by_flux.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved {png_path}")
    return fig


def main():
    datasets = {
        dataset_label: load_dataset(dataset_label, folder)
        for dataset_label, folder in DATASETS.items()
    }

    for dataset_label, dataset in datasets.items():
        print(f"Using {dataset_label} data from {dataset['csv_path']}")

    for metric_key, y_label, output_stem in METRICS:
        fig = plot_metric(datasets, metric_key, y_label, output_stem)
        plt.close(fig)


if __name__ == "__main__":
    main()

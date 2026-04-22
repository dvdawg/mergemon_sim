import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FLUXES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "sweep_results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"
CSV_GLOB = "imet_alpha_chi_vs_beta_flux*.csv"
RESULTS_DIR = DEFAULT_RESULTS_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
FLUX_VALUES = DEFAULT_FLUXES
EXCLUDED_FOLDERS = ["Ltot_075"]
SHOW_PLOT = False

# Choose "fixed" for a beta slice or "max-abs-chi" for the envelope over beta.
BETA_MODE = "fixed"
BETA_VALUE = 0.6


def parse_ltot_nh(folder_name):
    if not folder_name.startswith("Ltot_"):
        raise ValueError(f"Unexpected folder name: {folder_name}")

    suffix = folder_name.split("_", maxsplit=1)[1]
    if not suffix.isdigit():
        raise ValueError(f"Could not parse Ltot from {folder_name}")

    return int(suffix) / (10 ** (len(suffix) - 1))


def find_sweep_csv(folder):
    matches = sorted(
        path
        for path in folder.glob(CSV_GLOB)
        if "crossings" not in path.name
    )
    if not matches:
        raise FileNotFoundError(f"No sweep CSV found in {folder}")
    return matches[0]


def chi_mhz_from_row(row):
    """Sweep CSVs use chi_GHz; older runs may have chi_MHz. Normalize to MHz for plotting."""
    if "chi_MHz" in row and row["chi_MHz"] != "":
        return float(row["chi_MHz"])
    if "chi_GHz" in row and row["chi_GHz"] != "":
        return float(row["chi_GHz"]) * 1e3
    raise KeyError(
        "Expected a chi column: chi_GHz (preferred) or chi_MHz; "
        f"got keys {sorted(row.keys())!r}"
    )


def load_rows(csv_path):
    rows = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "beta": float(row["beta"]),
                    "phi": float(row["phi_ext_over_phi0"]),
                    "chi_mhz": chi_mhz_from_row(row),
                }
            )

    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")
    return rows


def unique_sorted(values):
    return np.array(sorted(set(values)), dtype=float)


def nearest_value(values, target):
    idx = int(np.argmin(np.abs(values - target)))
    return float(values[idx]), idx


def select_row_for_flux(rows, target_flux, beta_mode, beta_value):
    phi_values = unique_sorted(row["phi"] for row in rows)
    phi_used, _ = nearest_value(phi_values, target_flux)
    flux_rows = [row for row in rows if np.isclose(row["phi"], phi_used)]
    if not flux_rows:
        raise ValueError(f"No rows found for flux {phi_used}")

    finite_flux_rows = [
        row for row in flux_rows if np.isfinite(row["chi_mhz"])
    ]
    if not finite_flux_rows:
        raise ValueError(f"No finite chi values found for flux {phi_used}")

    if beta_mode == "max-abs-chi":
        selected = max(finite_flux_rows, key=lambda row: abs(row["chi_mhz"]))
    else:
        beta_values = unique_sorted(row["beta"] for row in finite_flux_rows)
        beta_used, _ = nearest_value(beta_values, beta_value)
        beta_rows = [
            row for row in finite_flux_rows if np.isclose(row["beta"], beta_used)
        ]
        if not beta_rows:
            raise ValueError(f"No rows found for beta {beta_used}")
        selected = beta_rows[0]

    return {
        "flux_requested": float(target_flux),
        "flux_used": float(phi_used),
        "beta_selected": float(selected["beta"]),
        "chi_mhz": float(selected["chi_mhz"]),
    }


def collect_dataset(results_dir, excluded_folders, flux_values, beta_mode, beta_value):
    dataset = []
    excluded = set(excluded_folders)

    for folder in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        if folder.name in excluded:
            continue
        if not folder.name.startswith("Ltot_"):
            continue

        ltot_nh = parse_ltot_nh(folder.name)
        csv_path = find_sweep_csv(folder)
        rows = load_rows(csv_path)

        flux_points = []
        for flux in flux_values:
            flux_points.append(
                select_row_for_flux(
                    rows=rows,
                    target_flux=flux,
                    beta_mode=beta_mode,
                    beta_value=beta_value,
                )
            )

        dataset.append(
            {
                "folder": folder.name,
                "csv_path": csv_path,
                "ltot_nh": ltot_nh,
                "ltot_h": ltot_nh * 1e-9,
                "flux_points": flux_points,
            }
        )

    dataset.sort(key=lambda item: item["ltot_nh"])
    if not dataset:
        raise ValueError(f"No usable Ltot folders found in {results_dir}")
    return dataset


def write_summary_csv(dataset, output_path):
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "folder",
                "source_csv",
                "Ltot_nH",
                "Ltot_H",
                "flux_requested",
                "flux_used",
                "beta_selected",
                "chi_MHz",
            ]
        )
        for item in dataset:
            for point in item["flux_points"]:
                writer.writerow(
                    [
                        item["folder"],
                        str(item["csv_path"]),
                        item["ltot_nh"],
                        item["ltot_h"],
                        point["flux_requested"],
                        point["flux_used"],
                        point["beta_selected"],
                        point["chi_mhz"],
                    ]
                )


def make_plot(dataset, beta_mode, beta_value, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    x_vals = [item["ltot_nh"] for item in dataset]

    flux_labels_seen = set()
    for flux_index in range(len(dataset[0]["flux_points"])):
        y_vals = [item["flux_points"][flux_index]["chi_mhz"] for item in dataset]
        flux_used = dataset[0]["flux_points"][flux_index]["flux_used"]
        label = f"flux = {flux_used:+.3f}"
        if label in flux_labels_seen:
            label = None
        else:
            flux_labels_seen.add(label)
        ax.plot(x_vals, y_vals, marker="o", linewidth=1.8, label=label)

    ax.set_xlabel(r"$L_\mathrm{tot}$ (nH)")
    ax.set_ylabel(r"$\chi$ (MHz)")
    if beta_mode == "fixed":
        title = rf"$\chi$ vs $L_\mathrm{{tot}}$ for selected flux biases ($\beta \approx {beta_value:.3f}$)"
    else:
        title = r"$\chi$ vs $L_\mathrm{tot}$ for selected flux biases (max $|\chi|$ over $\beta$)"
    ax.set_title(title)
    ax.set_xticks(x_vals)
    ax.grid(True, alpha=0.3)
    ax.legend(title=r"$\Phi_\mathrm{ext}/\Phi_0$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def main():
    dataset = collect_dataset(
        results_dir=RESULTS_DIR,
        excluded_folders=EXCLUDED_FOLDERS,
        flux_values=FLUX_VALUES,
        beta_mode=BETA_MODE,
        beta_value=BETA_VALUE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "chi_vs_ltot_by_flux.png"
    summary_path = OUTPUT_DIR / "chi_vs_ltot_by_flux_summary.csv"

    fig = make_plot(
        dataset=dataset,
        beta_mode=BETA_MODE,
        beta_value=BETA_VALUE,
        output_path=plot_path,
    )
    write_summary_csv(dataset, summary_path)

    print(f"Saved {plot_path}")
    print(f"Saved {summary_path}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

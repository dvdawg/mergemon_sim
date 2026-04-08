import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FLUXES = [-0.5, -0.25, 0.0, 0.25, 0.5]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "sweep_results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "plot_output"
CSV_GLOB = "imet_alpha_chi_vs_beta_flux*.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot chi versus total inductance Ltot using the sweep CSVs in "
            "imet_v1/sweep_results. Because each folder contains a beta/flux "
            "heatmap, the script collapses the beta axis either by selecting "
            "the beta with the largest |chi| at each flux or by taking a fixed "
            "beta slice."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing Ltot_* sweep folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the plot and summary CSV will be saved.",
    )
    parser.add_argument(
        "--flux-values",
        type=float,
        nargs="+",
        default=DEFAULT_FLUXES,
        help=(
            "Flux biases to plot in units of Phi_ext/Phi_0. The nearest sampled "
            "flux point in each CSV is used."
        ),
    )
    parser.add_argument(
        "--beta-mode",
        choices=("max-abs-chi", "fixed"),
        default="max-abs-chi",
        help=(
            "How to collapse the beta dimension inside each Ltot folder. "
            "'max-abs-chi' picks the beta with the largest |chi| for each flux. "
            "'fixed' uses the nearest available beta to --beta-value."
        ),
    )
    parser.add_argument(
        "--beta-value",
        type=float,
        default=0.5,
        help="Beta value used when --beta-mode=fixed.",
    )
    parser.add_argument(
        "--exclude-folder",
        action="append",
        default=["Ltot_075"],
        help="Folder name to ignore. May be passed multiple times.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving.",
    )
    return parser.parse_args()


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


def load_rows(csv_path):
    rows = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "beta": float(row["beta"]),
                    "phi": float(row["phi_ext_over_phi0"]),
                    "chi_mhz": float(row["chi_MHz"]),
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

    if beta_mode == "max-abs-chi":
        selected = max(flux_rows, key=lambda row: abs(row["chi_mhz"]))
    else:
        beta_values = unique_sorted(row["beta"] for row in flux_rows)
        beta_used, _ = nearest_value(beta_values, beta_value)
        flux_rows = [row for row in flux_rows if np.isclose(row["beta"], beta_used)]
        if not flux_rows:
            raise ValueError(f"No rows found for beta {beta_used}")
        selected = flux_rows[0]

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
    args = parse_args()
    dataset = collect_dataset(
        results_dir=args.results_dir,
        excluded_folders=args.exclude_folder,
        flux_values=args.flux_values,
        beta_mode=args.beta_mode,
        beta_value=args.beta_value,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "chi_vs_ltot_by_flux.png"
    summary_path = args.output_dir / "chi_vs_ltot_by_flux_summary.csv"

    fig = make_plot(
        dataset=dataset,
        beta_mode=args.beta_mode,
        beta_value=args.beta_value,
        output_path=plot_path,
    )
    write_summary_csv(dataset, summary_path)

    print(f"Saved {plot_path}")
    print(f"Saved {summary_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

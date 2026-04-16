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
# If None, use every distinct beta present in the sweep CSV (sorted).
BETA_VALUES = None
# Folder name under sweep_results, e.g. "Ltot_03"
LTOT_FOLDER_NAME = "Ltot_03"
SHOW_PLOT = False

# Mirror of plot_chi_vs_ltot_by_flux.BETA_MODE: slice flux or max |chi| over flux at fixed beta.
# "fixed": nearest phi to each curve's target flux, after narrowing to the requested beta.
# "max-abs-chi": all rows at phi matching the curve, then max |chi| over beta (independent of
#   the x-axis beta target; yields a horizontal segment per curve if you still sweep beta on x).
PHI_MODE = "fixed"
PHI_VALUE = 0.1


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


def select_row_for_beta(rows, target_beta, target_flux, phi_mode, phi_value):
    """Analog of select_row_for_flux: here the swept axis is beta, curves are flux."""
    if phi_mode == "max-abs-chi":
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

        selected = max(finite_flux_rows, key=lambda row: abs(row["chi_mhz"]))
        beta_values = unique_sorted(row["beta"] for row in finite_flux_rows)
        beta_used, _ = nearest_value(beta_values, target_beta)
        return {
            "beta_requested": float(target_beta),
            "beta_used": float(beta_used),
            "flux_requested": float(target_flux),
            "flux_used": float(phi_used),
            "chi_mhz": float(selected["chi_mhz"]),
        }

    beta_values = unique_sorted(row["beta"] for row in rows)
    beta_used, _ = nearest_value(beta_values, target_beta)
    beta_rows = [row for row in rows if np.isclose(row["beta"], beta_used)]
    if not beta_rows:
        raise ValueError(f"No rows found for beta {beta_used}")

    finite_beta_rows = [
        row for row in beta_rows if np.isfinite(row["chi_mhz"])
    ]
    if not finite_beta_rows:
        raise ValueError(f"No finite chi values found for beta {beta_used}")

    phi_values = unique_sorted(row["phi"] for row in finite_beta_rows)
    phi_used, _ = nearest_value(phi_values, phi_value)
    phi_rows = [
        row for row in finite_beta_rows if np.isclose(row["phi"], phi_used)
    ]
    if not phi_rows:
        raise ValueError(f"No rows found for phi {phi_used}")
    selected = phi_rows[0]

    return {
        "beta_requested": float(target_beta),
        "beta_used": float(beta_used),
        "flux_requested": float(target_flux),
        "flux_used": float(selected["phi"]),
        "chi_mhz": float(selected["chi_mhz"]),
    }


def collect_dataset(
    results_dir,
    ltot_folder_name,
    flux_values,
    beta_values,
    phi_mode,
    phi_value,
):
    folder = results_dir / ltot_folder_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing Ltot folder: {folder}")

    ltot_nh = parse_ltot_nh(folder.name)
    csv_path = find_sweep_csv(folder)
    rows = load_rows(csv_path)

    if beta_values is None:
        beta_list = unique_sorted(row["beta"] for row in rows)
    else:
        beta_list = np.array(beta_values, dtype=float)

    beta_points = []
    for target_beta in beta_list:
        flux_points = []
        for flux in flux_values:
            flux_points.append(
                select_row_for_beta(
                    rows=rows,
                    target_beta=float(target_beta),
                    target_flux=flux,
                    phi_mode=phi_mode,
                    phi_value=phi_value,
                )
            )
        beta_points.append({"flux_points": flux_points})

    return [
        {
            "folder": folder.name,
            "csv_path": csv_path,
            "ltot_nh": ltot_nh,
            "ltot_h": ltot_nh * 1e-9,
            "beta_points": beta_points,
        }
    ]


def write_summary_csv(dataset, output_path):
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "folder",
                "source_csv",
                "Ltot_nH",
                "Ltot_H",
                "beta_requested",
                "beta_used",
                "flux_requested",
                "flux_used",
                "chi_MHz",
            ]
        )
        item = dataset[0]
        for bp in item["beta_points"]:
            for point in bp["flux_points"]:
                writer.writerow(
                    [
                        item["folder"],
                        str(item["csv_path"]),
                        item["ltot_nh"],
                        item["ltot_h"],
                        point["beta_requested"],
                        point["beta_used"],
                        point["flux_requested"],
                        point["flux_used"],
                        point["chi_mhz"],
                    ]
                )


def make_plot(dataset, phi_mode, phi_value, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    item = dataset[0]
    beta_points = item["beta_points"]
    x_vals = [bp["flux_points"][0]["beta_used"] for bp in beta_points]

    flux_labels_seen = set()
    for flux_index in range(len(beta_points[0]["flux_points"])):
        y_vals = [
            bp["flux_points"][flux_index]["chi_mhz"] for bp in beta_points
        ]
        flux_used = beta_points[0]["flux_points"][flux_index]["flux_used"]
        label = f"flux = {flux_used:+.3f}"
        if label in flux_labels_seen:
            label = None
        else:
            flux_labels_seen.add(label)
        ax.plot(x_vals, y_vals, marker="o", linewidth=1.8, label=label)

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\chi$ (MHz)")
    ltot = item["ltot_nh"]
    if phi_mode == "fixed":
        title = (
            rf"$\chi$ vs $\beta$ for selected flux biases "
            rf"($L_\mathrm{{tot}}={ltot:.3g}$ nH, $\Phi/\Phi_0 \approx {phi_value:.3f}$)"
        )
    else:
        title = (
            rf"$\chi$ vs $\beta$ for selected flux biases "
            rf"($L_\mathrm{{tot}}={ltot:.3g}$ nH, max $|\chi|$ over $\beta$ at each flux)"
        )
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
        ltot_folder_name=LTOT_FOLDER_NAME,
        flux_values=FLUX_VALUES,
        beta_values=BETA_VALUES,
        phi_mode=PHI_MODE,
        phi_value=PHI_VALUE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_ltot = dataset[0]["folder"].replace(" ", "_")
    plot_path = OUTPUT_DIR / f"chi_vs_beta_by_flux_{safe_ltot}.png"
    summary_path = OUTPUT_DIR / f"chi_vs_beta_by_flux_{safe_ltot}_summary.csv"

    fig = make_plot(
        dataset=dataset,
        phi_mode=PHI_MODE,
        phi_value=PHI_VALUE,
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

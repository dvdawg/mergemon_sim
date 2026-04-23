import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FLUXES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_OUTPUT_SUBDIR = "plot_output"
CSV_GLOB = "imet_alpha_chi_vs_beta_flux*.csv"
FLUX_VALUES = DEFAULT_FLUXES
# If None, use every distinct beta present in the sweep CSV (sorted).
BETA_VALUES = None
# Total inductance in nanohenries. If None, choose a sensible default from the
# available sweep folders for the selected results directory.
LTOT_NH = 0.07
SHOW_PLOT = False

# Mirror of plot_chi_vs_ltot_by_flux.BETA_MODE (orthogonal to the swept axis).
# "fixed": snap beta to the x-axis target, then snap phi to each curve's target flux.
# "max-abs-chi": snap phi to each curve's target flux, then max |chi| over beta (same as the
#   L_tot script); chi is constant along the beta axis for a given curve (horizontal lines).
PHI_MODE = "fixed"


def default_results_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "sweep_results",
        script_dir / "sweep_results_10ghz_res",
        script_dir / "sweep_results_6.9ghz_res",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    available = sorted(
        path.name for path in script_dir.iterdir()
        if path.is_dir() and path.name.startswith("sweep_results")
    )
    raise FileNotFoundError(
        "Could not find a sweep results directory next to the script. "
        f"Looked for: {', '.join(str(path.name) for path in candidates)}. "
        f"Available sweep result directories: {', '.join(available) or '(none)'}"
    )


RESULTS_DIR = default_results_dir()


def parse_ltot_nh(folder_name):
    if not folder_name.startswith("Ltot_"):
        raise ValueError(f"Unexpected folder name: {folder_name}")

    suffix = folder_name.split("_", maxsplit=1)[1]
    if not suffix.isdigit():
        raise ValueError(f"Could not parse Ltot from {folder_name}")

    return int(suffix) / (10 ** (len(suffix) - 1))


def find_ltot_folder_by_nh(results_dir, ltot_nh):
    """Return sweep_results/Ltot_* whose parsed L_tot (nH) matches ``ltot_nh``."""
    target = float(ltot_nh)
    matches = []
    for path in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if not path.name.startswith("Ltot_"):
            continue
        try:
            parsed = parse_ltot_nh(path.name)
        except ValueError:
            continue
        if np.isclose(parsed, target, rtol=0.0, atol=1e-12):
            matches.append(path)

    if not matches:
        available = []
        for path in sorted(p for p in results_dir.iterdir() if p.is_dir()):
            if not path.name.startswith("Ltot_"):
                continue
            try:
                available.append(f"{path.name} ({parse_ltot_nh(path.name)} nH)")
            except ValueError:
                available.append(path.name)
        raise FileNotFoundError(
            f"No folder under {results_dir} matches L_tot={target} nH. "
            f"Available: {', '.join(available) or '(none)'}"
        )
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise ValueError(
            f"Multiple folders match L_tot={target} nH: {names}. "
            "Rename or remove duplicates."
        )
    return matches[0]


def default_ltot_nh(results_dir):
    available = sorted(
        parse_ltot_nh(path.name)
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith("Ltot_")
    )
    if not available:
        raise FileNotFoundError(f"No Ltot_* folders found in {results_dir}")

    preferred = [0.3, 0.03]
    for target in preferred:
        for value in available:
            if np.isclose(value, target, rtol=0.0, atol=1e-12):
                return value
    return available[0]


def find_sweep_csv(folder):
    matches = sorted(
        path
        for path in folder.glob(CSV_GLOB)
        if "crossings" not in path.name
    )
    if not matches:
        raise FileNotFoundError(f"No sweep CSV found in {folder}")
    return matches[0]


def output_dir_for_results(results_dir):
    return Path(results_dir).resolve() / DEFAULT_OUTPUT_SUBDIR


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


def select_row_for_beta(rows, target_beta, target_flux, phi_mode):
    """Analog of select_row_for_flux: swept axis is beta; curve parameter is flux (phi)."""
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
        beta_values = unique_sorted(row["beta"] for row in rows)
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
    phi_used, _ = nearest_value(phi_values, target_flux)
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
    ltot_nh,
    flux_values,
    beta_values,
    phi_mode,
):
    folder = find_ltot_folder_by_nh(results_dir, ltot_nh)
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


def make_plot(dataset, phi_mode, output_path):
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
            rf"($L_\mathrm{{tot}}={ltot:.3g}$ nH)"
        )
    else:
        title = (
            rf"$\chi$ vs $\beta$ for selected flux biases "
            rf"($L_\mathrm{{tot}}={ltot:.3g}$ nH, max $|\chi|$ over $\beta$)"
        )
    ax.set_title(title)
    ax.set_xticks(x_vals)
    ax.grid(True, alpha=0.3)
    ax.legend(title=r"$\Phi_\mathrm{ext}/\Phi_0$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def main():
    results_dir = Path(RESULTS_DIR).resolve()
    ltot_nh = LTOT_NH if LTOT_NH is not None else default_ltot_nh(results_dir)
    dataset = collect_dataset(
        results_dir=results_dir,
        ltot_nh=ltot_nh,
        flux_values=FLUX_VALUES,
        beta_values=BETA_VALUES,
        phi_mode=PHI_MODE,
    )

    output_dir = output_dir_for_results(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_ltot = dataset[0]["folder"].replace(" ", "_")
    plot_path = output_dir / f"chi_vs_beta_by_flux_{safe_ltot}.png"
    summary_path = output_dir / f"chi_vs_beta_by_flux_{safe_ltot}_summary.csv"

    fig = make_plot(
        dataset=dataset,
        phi_mode=PHI_MODE,
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

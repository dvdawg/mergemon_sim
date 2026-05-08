import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CSV_GLOB = "imet_alpha_chi_vs_beta_flux*.csv"
DEFAULT_BETA_FIXED = 0.2
DEFAULT_OUTPUT_SUBDIR = "plot_output/theory_vs_sim_dispersive_shift"
H = 6.62607015e-34
HBAR = 1.054571817e-34
PHI0 = 2.067833848e-15
ZERO_FLUX_TARGET = 0.0
SHOW_PLOT = False


def default_results_dirs():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "sweep_results_6.9ghz_res",
        script_dir / "sweep_results_10ghz_res",
    ]
    return [path for path in candidates if path.is_dir()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the dispersive-shift theory curve against simulated sweep CSVs "
            "using the zero-flux slice of the sweep data."
        )
    )
    parser.add_argument(
        "--results-dir",
        action="append",
        dest="results_dirs",
        help=(
            "Sweep-results directory to process. May be passed multiple times. "
            "Defaults to both sweep_results_6.9ghz_res and sweep_results_10ghz_res "
            "when present."
        ),
    )
    parser.add_argument(
        "--beta-fixed",
        type=float,
        default=DEFAULT_BETA_FIXED,
        help="Beta value used for the chi-vs-Z0r comparison.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively instead of closing them after saving.",
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


def parse_ltotsweep_vals(results_dir):
    table_path = Path(results_dir) / "Ltotsweep_vals.txt"
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing {table_path}")

    ltot_to_cr_pf = {}
    resonator_freq_ghz = None

    with table_path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("for resonator freq"):
                tail = line.split("=")[-1].split("≈")[-1].strip()
                resonator_freq_ghz = float(tail)
                continue

            pieces = line.split()
            if len(pieces) < 2:
                continue

            ltot_nh = float(pieces[0])
            c_r_pf = float(pieces[1])
            ltot_to_cr_pf[ltot_nh] = c_r_pf

    if not ltot_to_cr_pf:
        raise ValueError(f"No Ltot/C_r entries parsed from {table_path}")
    return ltot_to_cr_pf, resonator_freq_ghz


def chi_mhz_from_row(row):
    if "chi_MHz" in row and row["chi_MHz"] != "":
        return float(row["chi_MHz"])
    if "chi_GHz" in row and row["chi_GHz"] != "":
        return float(row["chi_GHz"]) * 1e3
    raise KeyError(
        "Expected chi_GHz or chi_MHz in CSV row; "
        f"got keys {sorted(row.keys())!r}"
    )


def omega_q_ghz_from_row(row):
    if "state_1_0_GHz" not in row:
        raise KeyError(
            "Expected state_1_0_GHz in CSV row so omega_q can be fixed."
        )
    e10 = float(row["state_1_0_GHz"])
    e00 = float(row.get("state_0_0_GHz", 0.0) or 0.0)
    return e10 - e00


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
                    "omega_q_ghz": omega_q_ghz_from_row(row),
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


def select_zero_flux_row_for_beta(rows, target_beta):
    beta_values = unique_sorted(row["beta"] for row in rows)
    beta_used, _ = nearest_value(beta_values, target_beta)
    beta_rows = [row for row in rows if np.isclose(row["beta"], beta_used)]
    if not beta_rows:
        raise ValueError(f"No rows found for beta {beta_used}")

    phi_values = unique_sorted(row["phi"] for row in beta_rows)
    phi_used, _ = nearest_value(phi_values, ZERO_FLUX_TARGET)
    phi_rows = [row for row in beta_rows if np.isclose(row["phi"], phi_used)]
    if not phi_rows:
        raise ValueError(f"No rows found for flux {phi_used}")

    selected = phi_rows[0]
    return {
        "beta_requested": float(target_beta),
        "beta_used": float(beta_used),
        "phi_requested": float(ZERO_FLUX_TARGET),
        "phi_used": float(phi_used),
        "omega_q_used_ghz": float(selected["omega_q_ghz"]),
        "chi_sim_mhz": float(selected["chi_mhz"]),
    }


def z0r_ohm_from_ltot_and_cr_pf(ltot_nh, c_r_pf):
    return np.sqrt((ltot_nh * 1e-9) / (c_r_pf * 1e-12))


def chi_theory_mhz(beta, omega_q_ghz, z0r_ohm):
    if beta <= 0.0 or beta >= 1.0:
        return np.nan
    l_ratio_sq = (beta / (1.0 - beta)) ** 2
    omega_q_rad_s = 2.0 * np.pi * omega_q_ghz * 1e9
    chi_rad_s = (
        -(H**2) / (32.0 * (PHI0**2) * HBAR)
        * l_ratio_sq
        * omega_q_rad_s
        * z0r_ohm
    )
    return chi_rad_s / (2.0 * np.pi * 1e6)


def build_beta_curve_for_folder(folder, c_r_pf):
    csv_path = find_sweep_csv(folder)
    rows = load_rows(csv_path)
    ltot_nh = parse_ltot_nh(folder.name)
    z0r_ohm = z0r_ohm_from_ltot_and_cr_pf(ltot_nh, c_r_pf)

    points = []
    for beta in unique_sorted(row["beta"] for row in rows):
        point = select_zero_flux_row_for_beta(rows, beta)
        point["chi_theory_mhz"] = float(
            chi_theory_mhz(
                beta=point["beta_used"],
                omega_q_ghz=point["omega_q_used_ghz"],
                z0r_ohm=z0r_ohm,
            )
        )
        points.append(point)

    return {
        "folder": folder.name,
        "csv_path": csv_path,
        "ltot_nh": float(ltot_nh),
        "c_r_pf": float(c_r_pf),
        "z0r_ohm": float(z0r_ohm),
        "points": points,
    }


def collect_beta_curves(results_dir):
    c_lookup, resonator_freq_ghz = parse_ltotsweep_vals(results_dir)
    curves = []

    for folder in sorted(path for path in Path(results_dir).iterdir() if path.is_dir()):
        if not folder.name.startswith("Ltot_"):
            continue

        ltot_nh = parse_ltot_nh(folder.name)
        if ltot_nh not in c_lookup:
            print(
                f"Skipping {folder} because Ltot={ltot_nh} nH "
                "was not found in Ltotsweep_vals.txt."
            )
            continue

        curves.append(
            build_beta_curve_for_folder(
                folder=folder,
                c_r_pf=c_lookup[ltot_nh],
            )
        )

    curves.sort(key=lambda item: item["z0r_ohm"])
    if not curves:
        raise ValueError(f"No usable Ltot folders found in {results_dir}")
    return curves, resonator_freq_ghz


def collect_z0r_points(results_dir, beta_fixed):
    c_lookup, resonator_freq_ghz = parse_ltotsweep_vals(results_dir)
    points = []

    for folder in sorted(path for path in Path(results_dir).iterdir() if path.is_dir()):
        if not folder.name.startswith("Ltot_"):
            continue

        ltot_nh = parse_ltot_nh(folder.name)
        if ltot_nh not in c_lookup:
            print(
                f"Skipping {folder} because Ltot={ltot_nh} nH "
                "was not found in Ltotsweep_vals.txt."
            )
            continue

        csv_path = find_sweep_csv(folder)
        rows = load_rows(csv_path)
        selected = select_zero_flux_row_for_beta(rows, beta_fixed)
        z0r_ohm = z0r_ohm_from_ltot_and_cr_pf(ltot_nh, c_lookup[ltot_nh])
        selected.update(
            {
                "folder": folder.name,
                "csv_path": csv_path,
                "ltot_nh": float(ltot_nh),
                "c_r_pf": float(c_lookup[ltot_nh]),
                "z0r_ohm": float(z0r_ohm),
                "chi_theory_mhz": float(
                    chi_theory_mhz(
                        beta=selected["beta_used"],
                        omega_q_ghz=selected["omega_q_used_ghz"],
                        z0r_ohm=z0r_ohm,
                    )
                ),
            }
        )
        points.append(selected)

    points.sort(key=lambda item: item["z0r_ohm"])
    if not points:
        raise ValueError(f"No usable Ltot folders found in {results_dir}")
    return points, resonator_freq_ghz


def output_dir_for_results(results_dir):
    return Path(results_dir).resolve() / DEFAULT_OUTPUT_SUBDIR


def sanitize_tag(value):
    return str(value).replace(".", "p")


def write_beta_summary(curves, output_path):
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "folder",
                "source_csv",
                "Ltot_nH",
                "C_r_pF",
                "Z0r_ohm",
                "beta_requested",
                "beta_used",
                "phi_requested",
                "phi_used",
                "omega_q_used_GHz",
                "chi_sim_MHz",
                "chi_theory_MHz",
            ]
        )
        for curve in curves:
            for point in curve["points"]:
                writer.writerow(
                    [
                        curve["folder"],
                        str(curve["csv_path"]),
                        curve["ltot_nh"],
                        curve["c_r_pf"],
                        curve["z0r_ohm"],
                        point["beta_requested"],
                        point["beta_used"],
                        point["phi_requested"],
                        point["phi_used"],
                        point["omega_q_used_ghz"],
                        point["chi_sim_mhz"],
                        point["chi_theory_mhz"],
                    ]
                )


def write_z0r_summary(points, output_path):
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "folder",
                "source_csv",
                "Ltot_nH",
                "C_r_pF",
                "Z0r_ohm",
                "beta_requested",
                "beta_used",
                "phi_requested",
                "phi_used",
                "omega_q_used_GHz",
                "chi_sim_MHz",
                "chi_theory_MHz",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point["folder"],
                    str(point["csv_path"]),
                    point["ltot_nh"],
                    point["c_r_pf"],
                    point["z0r_ohm"],
                    point["beta_requested"],
                    point["beta_used"],
                    point["phi_requested"],
                    point["phi_used"],
                    point["omega_q_used_ghz"],
                    point["chi_sim_mhz"],
                    point["chi_theory_mhz"],
                ]
            )


def make_beta_plot(curves, resonator_freq_ghz, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.05, 0.95, len(curves)))

    for color, curve in zip(colors, curves):
        x_vals = [point["beta_used"] for point in curve["points"]]
        y_sim = [point["chi_sim_mhz"] for point in curve["points"]]
        y_theory = [point["chi_theory_mhz"] for point in curve["points"]]
        label = (
            rf"$L_{{tot}}={curve['ltot_nh']:.3g}$ nH, "
            rf"$Z_{{0,r}}={curve['z0r_ohm']:.2f}\ \Omega$"
        )
        ax.plot(
            x_vals,
            y_theory,
            color=color,
            linewidth=2.0,
            label=f"{label} theory",
        )
        ax.plot(
            x_vals,
            y_sim,
            color=color,
            marker="o",
            linestyle="None",
            markersize=4.5,
            label=f"{label} sim",
        )

    ax.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax.set_ylabel(r"$\chi$ (MHz)")
    ax.set_title(
        r"$\chi$ vs $\beta$"
        + (
            rf" ($f_r \approx {resonator_freq_ghz:.3f}$ GHz)"
            if resonator_freq_ghz is not None
            else ""
        )
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def make_beta_error_plot(curves, resonator_freq_ghz, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.05, 0.95, len(curves)))

    for color, curve in zip(colors, curves):
        x_vals = [point["beta_used"] for point in curve["points"]]
        y_err = [
            point["chi_sim_mhz"] - point["chi_theory_mhz"]
            for point in curve["points"]
        ]
        label = (
            rf"$L_{{tot}}={curve['ltot_nh']:.3g}$ nH, "
            rf"$Z_{{0,r}}={curve['z0r_ohm']:.2f}\ \Omega$"
        )
        ax.plot(
            x_vals,
            y_err,
            color=color,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=label,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax.set_ylabel(r"$\chi_\mathrm{sim} - \chi_\mathrm{theory}$ (MHz)")
    ax.set_title(
        r"Error in $\chi$ vs $\beta$"
        + (
            rf" ($f_r \approx {resonator_freq_ghz:.3f}$ GHz)"
            if resonator_freq_ghz is not None
            else ""
        )
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def make_z0r_plot(points, resonator_freq_ghz, beta_fixed, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    x_vals = [point["z0r_ohm"] for point in points]
    y_sim = [point["chi_sim_mhz"] for point in points]
    y_theory = [point["chi_theory_mhz"] for point in points]

    ax.plot(x_vals, y_theory, linewidth=2.0, label="Theory")
    ax.plot(x_vals, y_sim, marker="o", linestyle="None", markersize=5, label="Simulation")

    for point in points:
        ax.annotate(
            point["folder"],
            (point["z0r_ohm"], point["chi_sim_mhz"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    ax.set_xlabel(r"$Z_{0,r} = \sqrt{L_\mathrm{tot} / C_r}$ ($\Omega$)")
    ax.set_ylabel(r"$\chi$ (MHz)")
    ax.set_title(
        rf"$\chi$ vs $Z_{{0,r}}$ at fixed $\beta \approx {beta_fixed:.3f}$ "
        + (
            rf" ($f_r \approx {resonator_freq_ghz:.3f}$ GHz)"
            if resonator_freq_ghz is not None
            else ""
        )
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def make_z0r_error_plot(points, resonator_freq_ghz, beta_fixed, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    x_vals = [point["z0r_ohm"] for point in points]
    y_err = [point["chi_sim_mhz"] - point["chi_theory_mhz"] for point in points]

    ax.plot(x_vals, y_err, marker="o", linewidth=1.8, markersize=5)

    for point, y_val in zip(points, y_err):
        ax.annotate(
            point["folder"],
            (point["z0r_ohm"], y_val),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"$Z_{0,r} = \sqrt{L_\mathrm{tot} / C_r}$ ($\Omega$)")
    ax.set_ylabel(r"$\chi_\mathrm{sim} - \chi_\mathrm{theory}$ (MHz)")
    ax.set_title(
        rf"Error in $\chi$ vs $Z_{{0,r}}$ at fixed $\beta \approx {beta_fixed:.3f}$ "
        + (
            rf" ($f_r \approx {resonator_freq_ghz:.3f}$ GHz)"
            if resonator_freq_ghz is not None
            else ""
        )
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def process_results_dir(results_dir, beta_fixed, show_plot):
    results_dir = Path(results_dir).resolve()
    output_dir = output_dir_for_results(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    beta_curves, beta_res_freq = collect_beta_curves(results_dir)
    z0r_points, z0r_res_freq = collect_z0r_points(results_dir, beta_fixed)
    beta_tag = sanitize_tag(f"{beta_fixed:.3f}")

    beta_plot_path = output_dir / "chi_vs_beta_zero_flux.png"
    beta_error_plot_path = output_dir / "chi_vs_beta_zero_flux_error.png"
    beta_summary_path = output_dir / "chi_vs_beta_zero_flux_summary.csv"
    z0r_plot_path = output_dir / f"chi_vs_z0r_beta_{beta_tag}_zero_flux.png"
    z0r_error_plot_path = output_dir / f"chi_vs_z0r_beta_{beta_tag}_zero_flux_error.png"
    z0r_summary_path = output_dir / f"chi_vs_z0r_beta_{beta_tag}_zero_flux_summary.csv"

    fig_beta = make_beta_plot(
        curves=beta_curves,
        resonator_freq_ghz=beta_res_freq,
        output_path=beta_plot_path,
    )
    fig_beta_error = make_beta_error_plot(
        curves=beta_curves,
        resonator_freq_ghz=beta_res_freq,
        output_path=beta_error_plot_path,
    )
    fig_z0r = make_z0r_plot(
        points=z0r_points,
        resonator_freq_ghz=z0r_res_freq,
        beta_fixed=beta_fixed,
        output_path=z0r_plot_path,
    )
    fig_z0r_error = make_z0r_error_plot(
        points=z0r_points,
        resonator_freq_ghz=z0r_res_freq,
        beta_fixed=beta_fixed,
        output_path=z0r_error_plot_path,
    )
    write_beta_summary(beta_curves, beta_summary_path)
    write_z0r_summary(z0r_points, z0r_summary_path)

    print(f"Saved {beta_plot_path}")
    print(f"Saved {beta_error_plot_path}")
    print(f"Saved {beta_summary_path}")
    print(f"Saved {z0r_plot_path}")
    print(f"Saved {z0r_error_plot_path}")
    print(f"Saved {z0r_summary_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig_beta)
        plt.close(fig_beta_error)
        plt.close(fig_z0r)
        plt.close(fig_z0r_error)


def main():
    args = parse_args()
    results_dirs = args.results_dirs or [str(path) for path in default_results_dirs()]
    if not results_dirs:
        raise FileNotFoundError(
            "No sweep results directories were found. Pass --results-dir explicitly."
        )

    for results_dir in results_dirs:
        process_results_dir(
            results_dir=results_dir,
            beta_fixed=args.beta_fixed,
            show_plot=args.show or SHOW_PLOT,
        )


if __name__ == "__main__":
    main()

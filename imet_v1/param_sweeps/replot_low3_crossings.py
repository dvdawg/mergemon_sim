#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd


DEFAULT_LEVELS = (1, 2, 3)
DEFAULT_OUTPUT_DIRNAME = "replotted_crossings"
UNIT_SCALE = {"GHz": 1_000.0, "MHz": 1.0}
DISPLAY_UNIT = "MHz"
CHI_DISPLAY_ABS_MAX_MHZ = 100.0


def default_results_dir() -> Path:
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
        f"Looked for: {', '.join(path.name for path in candidates)}. "
        f"Available sweep result directories: {', '.join(available) or '(none)'}"
    )


def find_matching_csv(folder: Path, pattern: str, *, exclude_crossings: bool) -> Path:
    matches = sorted(
        path
        for path in folder.glob(pattern)
        if ("crossings" not in path.name) == exclude_crossings
    )
    if not matches:
        kind = "sweep" if exclude_crossings else "crossings"
        raise FileNotFoundError(f"No {kind} CSV found in {folder}")
    return matches[0]


def reshape_grid(df: pd.DataFrame, column: str, beta_vals: np.ndarray, phi_vals: np.ndarray) -> np.ndarray:
    grid = (
        df.pivot(index="phi_ext_over_phi0", columns="beta", values=column)
        .reindex(index=phi_vals, columns=beta_vals)
        .to_numpy(dtype=float)
    )
    return grid


def axis_edges(vals: np.ndarray) -> tuple[float, float]:
    if vals.size == 1:
        return float(vals[0] - 0.5), float(vals[0] + 0.5)
    diffs = np.diff(vals)
    first = vals[0] - 0.5 * diffs[0]
    last = vals[-1] + 0.5 * diffs[-1]
    return float(first), float(last)


def resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"None of the expected columns were found: {', '.join(candidates)}")


def unit_from_column(column: str) -> str:
    if column.endswith("_GHz"):
        return "GHz"
    if column.endswith("_MHz"):
        return "MHz"
    return "arb."


def scale_to_display_units(values: np.ndarray, source_unit: str) -> np.ndarray:
    scale = UNIT_SCALE.get(source_unit)
    if scale is None:
        return values
    return values * scale


def filtered_crossings_df(path: Path, low_levels: tuple[int, ...], filter_mode: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    low_level_set = set(low_levels)
    if filter_mode == "within":
        keep = df["level_a"].isin(low_level_set) & df["level_b"].isin(low_level_set)
    elif filter_mode == "involving":
        keep = df["level_a"].isin(low_level_set) | df["level_b"].isin(low_level_set)
    else:
        raise ValueError(f"Unsupported filter mode: {filter_mode}")
    return df.loc[keep].copy()


def format_levels(levels: tuple[int, ...]) -> str:
    return ", ".join(str(level) for level in levels)


def output_name_for_folder(folder: Path) -> str:
    return f"{folder.name}_low3_crossings.png"


def plot_folder(folder: Path, output_dir: Path, low_levels: tuple[int, ...], filter_mode: str) -> Path:
    sweep_csv = find_matching_csv(folder, "imet_alpha_chi_vs_beta_flux*.csv", exclude_crossings=True)
    crossings_csv = find_matching_csv(
        folder, "imet_alpha_chi_vs_beta_flux_crossings*.csv", exclude_crossings=False
    )

    sweep_df = pd.read_csv(sweep_csv)
    crossings_df = filtered_crossings_df(crossings_csv, low_levels, filter_mode)

    alpha_column = resolve_column(sweep_df, ("alpha_GHz", "alpha_MHz"))
    chi_column = resolve_column(sweep_df, ("chi_GHz", "chi_MHz"))
    abs_chi_column = resolve_column(sweep_df, ("abs_chi_GHz", "abs_chi_MHz"))
    source_units = unit_from_column(alpha_column)

    beta_vals = np.array(sorted(sweep_df["beta"].unique()), dtype=float)
    phi_vals = np.array(sorted(sweep_df["phi_ext_over_phi0"].unique()), dtype=float)
    beta_min, beta_max = axis_edges(beta_vals)
    phi_min, phi_max = axis_edges(phi_vals)

    alpha_grid = scale_to_display_units(
        reshape_grid(sweep_df, alpha_column, beta_vals, phi_vals), source_units
    )
    chi_grid = scale_to_display_units(
        reshape_grid(sweep_df, chi_column, beta_vals, phi_vals), source_units
    )
    abs_chi_grid = scale_to_display_units(
        reshape_grid(sweep_df, abs_chi_column, beta_vals, phi_vals), source_units
    )

    if np.all(np.isnan(alpha_grid)):
        alpha_vmin, alpha_vmax = -1.0, 1.0
    else:
        alpha_vmin = float(np.nanmin(alpha_grid))
        alpha_vmax = float(np.nanmax(alpha_grid))

    chi_norm = None
    if np.all(np.isnan(chi_grid)):
        chi_vmin, chi_vmax = -1.0, 1.0
    else:
        chi_abs_max = min(float(np.nanmax(np.abs(chi_grid))), CHI_DISPLAY_ABS_MAX_MHZ)
        chi_vmin, chi_vmax = -chi_abs_max, chi_abs_max
        if chi_abs_max > 0.0:
            chi_linthresh = max(0.5, chi_abs_max * 0.02)
            chi_norm = colors.SymLogNorm(
                linthresh=chi_linthresh,
                linscale=1.0,
                vmin=chi_vmin,
                vmax=chi_vmax,
                base=10.0,
            )

    abs_chi_vmax = float(np.nanmax(abs_chi_grid)) if not np.all(np.isnan(abs_chi_grid)) else 1.0

    fig, (ax_alpha, ax_chi, ax_abs_chi) = plt.subplots(1, 3, figsize=(20, 5))

    im_alpha = ax_alpha.pcolormesh(
        beta_vals, phi_vals, alpha_grid, shading="auto", cmap="viridis", vmin=alpha_vmin, vmax=alpha_vmax
    )
    ax_alpha.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_alpha.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_alpha.set_title(rf"Anharmonicity $\alpha$ ({DISPLAY_UNIT})")
    plt.colorbar(im_alpha, ax=ax_alpha, label=rf"$\alpha$ ({DISPLAY_UNIT})")

    im_chi = ax_chi.pcolormesh(
        beta_vals,
        phi_vals,
        chi_grid,
        shading="auto",
        cmap="plasma",
        norm=chi_norm,
        vmin=None if chi_norm is not None else chi_vmin,
        vmax=None if chi_norm is not None else chi_vmax,
    )
    ax_chi.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_chi.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_chi.set_title(rf"Dispersive shift $\chi$ ({DISPLAY_UNIT})")
    plt.colorbar(im_chi, ax=ax_chi, label=rf"$\chi$ ({DISPLAY_UNIT})")

    im_abs_chi = ax_abs_chi.pcolormesh(
        beta_vals,
        phi_vals,
        abs_chi_grid,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=abs_chi_vmax,
    )
    ax_abs_chi.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_abs_chi.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_abs_chi.set_title(rf"Magnitude of dispersive shift $|\chi|$ ({DISPLAY_UNIT})")
    plt.colorbar(im_abs_chi, ax=ax_abs_chi, label=rf"$|\chi|$ ({DISPLAY_UNIT})")

    if not crossings_df.empty:
        beta_cross = crossings_df["beta"].to_numpy(dtype=float)
        phi_cross = crossings_df["phi_ext_over_phi0"].to_numpy(dtype=float)
        kinds = crossings_df["kind"].astype(str).to_numpy()

        crossing_mask = kinds == "crossing"
        avoided_mask = kinds == "avoided"

        if np.any(crossing_mask):
            ax_alpha.scatter(
                beta_cross[crossing_mask],
                phi_cross[crossing_mask],
                s=34,
                marker="o",
                facecolors="none",
                edgecolors="w",
                linewidths=1.0,
                zorder=8,
                clip_on=False,
            )
            ax_chi.scatter(
                beta_cross[crossing_mask],
                phi_cross[crossing_mask],
                s=34,
                marker="o",
                facecolors="none",
                edgecolors="w",
                linewidths=1.0,
                zorder=8,
                clip_on=False,
            )
            ax_abs_chi.scatter(
                beta_cross[crossing_mask],
                phi_cross[crossing_mask],
                s=34,
                marker="o",
                facecolors="none",
                edgecolors="w",
                linewidths=1.0,
                zorder=8,
                clip_on=False,
            )

        if np.any(avoided_mask):
            ax_alpha.scatter(
                beta_cross[avoided_mask],
                phi_cross[avoided_mask],
                s=24,
                marker="x",
                c="w",
                linewidths=1.0,
                zorder=8,
                clip_on=False,
            )
            ax_chi.scatter(
                beta_cross[avoided_mask],
                phi_cross[avoided_mask],
                s=24,
                marker="x",
                c="w",
                linewidths=1.0,
                zorder=8,
                clip_on=False,
            )
            ax_abs_chi.scatter(
                beta_cross[avoided_mask],
                phi_cross[avoided_mask],
                s=24,
                marker="x",
                c="w",
                linewidths=1.0,
                zorder=8,
                clip_on=False,
            )

    for ax in (ax_alpha, ax_chi, ax_abs_chi):
        ax.set_aspect("auto")
        ax.tick_params(axis="both", labelsize=10)
        ax.set_xlim(beta_min, beta_max)
        ax.set_ylim(phi_min, phi_max)

    if filter_mode == "within":
        title_suffix = f"crossings within tracked levels {format_levels(low_levels)}"
    else:
        title_suffix = f"crossings involving tracked levels {format_levels(low_levels)}"

    fig.suptitle(
        rf"$\alpha$, $\chi$, and $|\chi|$ vs $\beta$ and flux ({folder.name}; {title_suffix})",
        fontsize=14,
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name_for_folder(folder)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot beta/flux heatmaps with crossings filtered by tracked energy level index."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_results_dir(),
        help="Directory containing the per-sweep result folders.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="Tracked level indices to highlight. Default: 1 2 3",
    )
    parser.add_argument(
        "--filter-mode",
        choices=("involving", "within"),
        default="within",
        help=(
            "'involving' keeps crossings where either level is in --levels; "
            "'within' keeps only crossings where both levels are in --levels."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for replotted PNGs. Default: <root>/replotted_crossings",
    )
    args = parser.parse_args()

    low_levels = tuple(sorted(set(args.levels)))
    output_dir = args.output_dir if args.output_dir is not None else args.root / DEFAULT_OUTPUT_DIRNAME
    folders = sorted(path for path in args.root.glob("Ltot_*") if path.is_dir())
    for folder in folders:
        out_path = plot_folder(folder, output_dir, low_levels, args.filter_mode)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

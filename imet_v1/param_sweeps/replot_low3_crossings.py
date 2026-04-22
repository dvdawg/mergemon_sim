#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_NAME = "imet_alpha_chi_vs_beta_flux_low3_crossings.png"
DEFAULT_LEVELS = (1, 2, 3)


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


def plot_folder(folder: Path, low_levels: tuple[int, ...], filter_mode: str) -> Path:
    sweep_csv = find_matching_csv(folder, "imet_alpha_chi_vs_beta_flux*.csv", exclude_crossings=True)
    crossings_csv = find_matching_csv(
        folder, "imet_alpha_chi_vs_beta_flux_crossings*.csv", exclude_crossings=False
    )

    sweep_df = pd.read_csv(sweep_csv)
    crossings_df = filtered_crossings_df(crossings_csv, low_levels, filter_mode)

    beta_vals = np.array(sorted(sweep_df["beta"].unique()), dtype=float)
    phi_vals = np.array(sorted(sweep_df["phi_ext_over_phi0"].unique()), dtype=float)
    beta_min, beta_max = axis_edges(beta_vals)
    phi_min, phi_max = axis_edges(phi_vals)

    alpha_mhz = reshape_grid(sweep_df, "alpha_MHz", beta_vals, phi_vals)
    chi_mhz = reshape_grid(sweep_df, "chi_MHz", beta_vals, phi_vals)
    abs_chi_mhz = reshape_grid(sweep_df, "abs_chi_MHz", beta_vals, phi_vals)

    if np.all(np.isnan(alpha_mhz)):
        alpha_vmin, alpha_vmax = -1.0, 1.0
    else:
        alpha_vmin = float(np.nanmin(alpha_mhz))
        alpha_vmax = float(np.nanmax(alpha_mhz))

    if np.all(np.isnan(chi_mhz)):
        chi_vmin, chi_vmax = -1.0, 1.0
    else:
        chi_abs_max = float(np.nanmax(np.abs(chi_mhz)))
        chi_vmin, chi_vmax = -chi_abs_max, chi_abs_max

    fig, (ax_alpha, ax_chi, ax_abs_chi) = plt.subplots(1, 3, figsize=(20, 5))

    im_alpha = ax_alpha.pcolormesh(
        beta_vals, phi_vals, alpha_mhz, shading="auto", cmap="viridis", vmin=alpha_vmin, vmax=alpha_vmax
    )
    ax_alpha.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_alpha.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_alpha.set_title(r"Anharmonicity $\alpha$ (MHz)")
    plt.colorbar(im_alpha, ax=ax_alpha, label=r"$\alpha$ (MHz)")

    im_chi = ax_chi.pcolormesh(
        beta_vals, phi_vals, chi_mhz, shading="auto", cmap="plasma", vmin=chi_vmin, vmax=chi_vmax
    )
    ax_chi.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_chi.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_chi.set_title(r"Dispersive shift $\chi$ (MHz)")
    plt.colorbar(im_chi, ax=ax_chi, label=r"$\chi$ (MHz)")

    im_abs_chi = ax_abs_chi.pcolormesh(
        beta_vals,
        phi_vals,
        abs_chi_mhz,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=float(np.nanmax(abs_chi_mhz)) if not np.all(np.isnan(abs_chi_mhz)) else 1.0,
    )
    ax_abs_chi.set_xlabel(r"$\beta = L_c / L_\mathrm{tot}$")
    ax_abs_chi.set_ylabel(r"External flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax_abs_chi.set_title(r"Magnitude of dispersive shift $|\chi|$ (MHz)")
    plt.colorbar(im_abs_chi, ax=ax_abs_chi, label=r"$|\chi|$ (MHz)")

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

    fig.suptitle(rf"$\alpha$, $\chi$, and $|\chi|$ vs $\beta$ and flux ({title_suffix})", fontsize=14)
    fig.tight_layout()

    out_path = folder / OUTPUT_NAME
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
        default=Path(__file__).resolve().parent / "sweep_results",
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
        default="involving",
        help=(
            "'involving' keeps crossings where either level is in --levels; "
            "'within' keeps only crossings where both levels are in --levels."
        ),
    )
    args = parser.parse_args()

    low_levels = tuple(sorted(set(args.levels)))
    folders = sorted(path for path in args.root.iterdir() if path.is_dir())
    for folder in folders:
        out_path = plot_folder(folder, low_levels, args.filter_mode)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

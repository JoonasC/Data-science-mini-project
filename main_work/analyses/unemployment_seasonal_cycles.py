"""Produce two PNGs: monthly deviations and a combined annual floor plot.

Requires: numpy, pandas, matplotlib.
Input columns: Month (YYYYMmm), ISCO, Occupation, Unemployed jobseekers.

Run after map_unemployment_by_occupation_to_monthly.py; edit the paths below, or:
    python unemployment_seasonal_cycles.py --input monthly.csv --output-dir plots

For each occupation:
    trend = centred 2x12-month moving average
    deviation = 100 * (monthly unemployment / trend - 1)
    plotted deviation = centred three-month mean of deviation

The annual average uses 13 weights: 1/24 at each end and 1/12 for the
11 interior months. SMOOTH_MONTHS=1 disables the short additional smoothing.
This is the same deviation signal as the previous diagnostic plot.

Recurring annual movement is visible, but this signal also contains irregular
shocks. A centred filter spreads sudden changes into surrounding months, so
the dips adjoining the COVID spike need not be separate labour-market events.
The filter removes slow changes: a stable deviation pattern does not rule out
a rise in the underlying unemployment level. Both filters use future months,
making this retrospective analysis rather than a forecasting test.

The deviation y-label is 'Deviation from floor (%)', as requested. Its
denominator remains the centred annual moving average described above.
The separate floor graph uses each complete calendar year's minimum monthly
count, indexed to BASE_YEAR=100 for comparison between occupations.

Outputs only: unemployment_seasonal_deviations.png and unemployment_floor.png.
Missing values at the edges reflect the observations needed by the filters.
"""
import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


INPUT_PATH = Path("../../data_processed/Unemployment by month ISCO 1-digit.csv")
OUTPUT_DIR = Path(".")
VALUE_COL = "Unemployed jobseekers"
SMOOTH_MONTHS = 3                   # odd; 1 disables additional smoothing
COVID_START, COVID_END = 2020, 2022
POST_START = 2023
BASE_YEAR = 2019                    # annual minimum count index = 100


def read_data(path):
    df = pd.read_csv(path)
    required = {"Month", "ISCO", "Occupation", VALUE_COL}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing input columns: {sorted(required - set(df.columns))}")
    df["Date"] = pd.to_datetime(df["Month"], format="%YM%m", errors="raise")
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="raise")
    if df[["Date", "ISCO", "Occupation", VALUE_COL]].isna().any().any():
        raise ValueError("Input has missing dates, identifiers or unemployment counts.")
    if not np.isfinite(df[VALUE_COL]).all() or (df[VALUE_COL] <= 0).any():
        raise ValueError("Relative deviations require finite, positive counts.")
    if df.duplicated(["Date", "ISCO"]).any():
        raise ValueError("Duplicate month/ISCO rows in input.")
    df = df[~df["ISCO"].isin(["OC0", "OC6"])].copy()
    if df.empty:
        raise ValueError("No retained ISCO groups found.")
    for isco, d in df.groupby("ISCO"):
        expected = pd.date_range(d.Date.min(), d.Date.max(), freq="MS")
        if len(d) != len(expected):
            raise ValueError(f"{isco}: missing calendar months; check the source.")
        if d.Occupation.nunique() != 1:
            raise ValueError(f"{isco}: inconsistent occupation labels.")
    return df.sort_values(["ISCO", "Date"])


def calculate_deviations(df):
    groups = []
    weights = np.r_[0.5, np.ones(11), 0.5] / 12
    for _, d in df.groupby("ISCO", sort=True):
        d = d.copy()
        d["Annual moving average"] = d[VALUE_COL].rolling(13, center=True).apply(
            lambda x: np.dot(x, weights), raw=True
        )
        d["Deviation (%)"] = 100 * (d[VALUE_COL] / d["Annual moving average"] - 1)
        d["Smoothed deviation (%)"] = d["Deviation (%)"].rolling(
            SMOOTH_MONTHS, center=True
        ).mean()
        groups.append(d)
    return pd.concat(groups, ignore_index=True)


def add_period_legend(fig):
    fig.legend(
        handles=[
            Patch(facecolor="grey", alpha=0.12, label="COVID period"),
            Patch(facecolor="#e8b757", alpha=0.10, label="Generative AI adoption period"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=2, frameon=False, fontsize=10,
    )


def plot_deviations(df, output):
    groups = sorted(df.ISCO.unique())
    names = df.drop_duplicates("ISCO").set_index("ISCO").Occupation
    colors = dict(zip(groups, plt.cm.tab10(np.linspace(0, 1, 10))[:len(groups)]))
    rows = int(np.ceil(len(groups) / 2))
    fig, axes = plt.subplots(
        rows, 2, figsize=(17, rows * 2.65 + 1.5), sharex=True, squeeze=False
    )
    start, end = df.Date.min(), df.Date.max()
    for ax, isco in zip(axes.flat, groups):
        d = df[df.ISCO == isco]
        ax.plot(d.Date, d["Smoothed deviation (%)"], color=colors[isco], linewidth=1.3)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axvspan(pd.Timestamp(COVID_START, 1, 1), pd.Timestamp(COVID_END + 1, 1, 1),
                   color="grey", alpha=0.12)
        if end >= pd.Timestamp(POST_START, 1, 1):
            ax.axvspan(pd.Timestamp(POST_START, 1, 1), end,
                       color="#e8b757", alpha=0.10)
        ax.set_title(f"{isco}: {names[isco]}", fontsize=10, loc="left")
        ax.set_ylabel("Deviation from annual\nmoving average (%)")
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.suptitle("Seasonal unemployment cycles",
                 fontsize=15, y=0.99)
    add_period_legend(fig)
    fig.tight_layout(rect=[0, 0.055, 1, 1], h_pad=1.5)
    fig.savefig(output / "unemployment_seasonal_deviations.png", dpi=150)


def calculate_floors(df):
    """Annual minima for complete calendar years, indexed within each group."""
    annual = (
        df.assign(Year=df.Date.dt.year)
        .groupby(["Year", "ISCO", "Occupation"], as_index=False)
        .agg(Floor=(VALUE_COL, "min"), Months=("Date", "nunique"))
    )
    annual = annual[annual.Months.eq(12)].copy()
    base = annual[annual.Year.eq(BASE_YEAR)].set_index("ISCO").Floor
    missing = sorted(set(df.ISCO) - set(base.index))
    if missing:
        raise ValueError(f"A complete {BASE_YEAR} is required for floor indexing: {missing}")
    annual["Floor index"] = 100 * annual.Floor / annual.ISCO.map(base)
    return annual.sort_values(["ISCO", "Year"])


def plot_floor(annual, output):
    groups = sorted(annual.ISCO.unique())
    names = annual.drop_duplicates("ISCO").set_index("ISCO").Occupation
    colors = dict(zip(groups, plt.cm.tab10(np.linspace(0, 1, 10))[:len(groups)]))
    fig, ax = plt.subplots(figsize=(17, 7))
    lo, hi = int(annual.Year.min()), int(annual.Year.max())
    ax.axvspan(COVID_START - 0.5, COVID_END + 0.5, color="grey", alpha=0.12)
    if hi >= POST_START:
        ax.axvspan(POST_START - 0.5, hi + 0.5, color="#e8b757", alpha=0.10)
    for isco, d in annual.groupby("ISCO", sort=True):
        ax.plot(d.Year, d["Floor index"], color=colors[isco], alpha=0.5, linewidth=1)
        ax.scatter(d.Year, d["Floor index"], color=colors[isco], s=55, zorder=3,
                   label=f"{isco}: {names[isco]}")
    ax.axhline(100, color="grey", linestyle=":", linewidth=1)
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Unemployment floor (index, {BASE_YEAR} = 100)")
    ax.set_title("Annual minimum unemployed jobseekers by ISCO occupation group", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=False)
    add_period_legend(fig)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output / "unemployment_floor.png", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if SMOOTH_MONTHS < 1 or SMOOTH_MONTHS % 2 != 1:
        raise ValueError("SMOOTH_MONTHS must be a positive odd integer.")
    df = calculate_deviations(read_data(args.input))
    if df["Smoothed deviation (%)"].notna().sum() == 0:
        raise ValueError("Too few months to calculate the centred deviations.")
    annual = calculate_floors(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_deviations(df, args.output_dir)
    plot_floor(annual, args.output_dir)
    print(f"Wrote the two PNG images to {args.output_dir.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()

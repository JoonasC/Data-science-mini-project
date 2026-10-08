from pathlib import Path

import numpy as np
import pandas as pd


EMPLOYMENT_PATH = Path("../data/Eurostat employment by NACE and ISCO.csv")
AI_PATH = Path("../data/AI usage in enterprises by year.csv")
OUTPUT_PATH = Path("../data_processed/AI usage in enterprises by year ISCO 1-digit.csv")

# Use a single 2021-2025 average employment structure to reduce year-to-year
# sampling noise in the EU-LFS NACE x ISCO cells.
EMPLOYMENT_START_YEAR = 2021
EMPLOYMENT_END_YEAR = 2025

# OC0 (Armed forces) and OC6 (Skilled agricultural/forestry/fishery workers)
# are intentionally excluded.
ISCO_TO_KEEP = ["OC1", "OC2", "OC3", "OC4", "OC5", "OC7", "OC8", "OC9"]

# Match the broad industry groups available in the AI usage CSV.
# S951 is not available separately in the Eurostat employment file, so the
# L,N,S951 AI category is approximated with employment in L + N.
NACE_TO_AI_GROUP = {
    "C": "CDE",
    "D": "CDE",
    "E": "CDE",
    "F": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "J": "J",
    "L": "LN",
    "M": "M",
    "N": "LN",
}


def classify_ai_industry(industry: str) -> str | None:
    """Map a Statistics Finland AI-industry label to our common industry groups."""
    if industry.startswith("C-E "):
        return "CDE"
    if industry.startswith("41-43 "):
        return "F"
    if industry.startswith("45-46 ") or industry.startswith("47 "):
        return "G"
    if industry.startswith("H "):
        return "H"
    if industry.startswith("55, 56 "):
        return "I"
    if industry.startswith("J "):
        return "J"
    if industry.startswith("L, N, S951 "):
        return "LN"
    if industry.startswith("M "):
        return "M"
    return None


def load_employment_weights(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    employment = pd.read_csv(path, na_values=[".", ":", ""])

    employment["TIME_PERIOD"] = pd.to_numeric(employment["TIME_PERIOD"], errors="coerce")
    employment["OBS_VALUE"] = pd.to_numeric(employment["OBS_VALUE"], errors="coerce")

    employment = employment[
        employment["TIME_PERIOD"].between(EMPLOYMENT_START_YEAR, EMPLOYMENT_END_YEAR)
        & employment["isco08"].isin(ISCO_TO_KEEP)
        & employment["nace_r2"].isin(NACE_TO_AI_GROUP)
    ].copy()

    # Keep the published numeric values even when Eurostat marks them with u/bu.
    # Suppressed cells have no OBS_VALUE and therefore remain NaN.
    employment["ai_group"] = employment["nace_r2"].map(NACE_TO_AI_GROUP)

    isco_labels = (
        employment.dropna(subset=["International Standard Classification of Occupations 2008 (ISCO-08)"])
        .drop_duplicates("isco08")
        .set_index("isco08")["International Standard Classification of Occupations 2008 (ISCO-08)"]
        .to_dict()
    )

    # First aggregate constituent NACE sections within each year, e.g. C + D + E.
    yearly = (
        employment.groupby(["TIME_PERIOD", "isco08", "ai_group"], as_index=False)["OBS_VALUE"]
        .sum(min_count=1)
    )

    # Then average 2021-2025 to obtain a stable employment structure.
    avg_employment = (
        yearly.groupby(["isco08", "ai_group"], as_index=False)["OBS_VALUE"]
        .mean()
        .rename(columns={"OBS_VALUE": "avg_employment_thousand"})
    )

    # For each ISCO group, weights sum to 1 across the AI-covered industry groups.
    avg_employment["weight"] = avg_employment.groupby("isco08")[
        "avg_employment_thousand"
    ].transform(lambda x: x / x.sum())

    return avg_employment, isco_labels


def load_ai_by_group(path: Path) -> tuple[pd.DataFrame, list[str]]:
    ai = pd.read_csv(path, na_values=[".", ":", ""])
    ai["Year"] = pd.to_numeric(ai["Year"], errors="coerce").astype("Int64")
    ai["ai_group"] = ai["Industry"].map(classify_ai_industry)
    ai = ai[ai["ai_group"].notna()].copy()

    metric_columns = [c for c in ai.columns if c not in {"Year", "Industry", "ai_group"}]
    ai[metric_columns] = ai[metric_columns].apply(pd.to_numeric, errors="coerce")

    # The AI file splits NACE G into 45-46 and 47, while lfsa_eisn2 only has G.
    # With only these two input files there is no correct enterprise-count weight
    # for combining the two percentages, so this uses their simple mean.
    # Replace this with an enterprise-count weighted mean if you add those counts.
    ai_grouped = (
        ai.groupby(["Year", "ai_group"], as_index=False)[metric_columns]
        .mean()
    )

    return ai_grouped, metric_columns


def map_ai_to_isco(
    ai_grouped: pd.DataFrame,
    metric_columns: list[str],
    weights: pd.DataFrame,
    isco_labels: dict[str, str],
) -> pd.DataFrame:
    weight_matrix = weights.pivot(index="isco08", columns="ai_group", values="weight")
    ai_by_year = {
        int(year): frame.set_index("ai_group")[metric_columns]
        for year, frame in ai_grouped.groupby("Year")
    }

    rows: list[dict[str, object]] = []

    for year in sorted(ai_by_year):
        year_ai = ai_by_year[year]

        for isco in ISCO_TO_KEEP:
            w = weight_matrix.loc[isco]
            row: dict[str, object] = {
                "Year": year,
                "ISCO": isco,
                "Occupation": isco_labels.get(isco, isco),
            }

            for metric in metric_columns:
                values = year_ai[metric].reindex(w.index)
                valid = values.notna() & w.notna()

                if not valid.any():
                    row[metric] = np.nan
                    continue

                # Renormalise over industries for which this AI metric is available.
                metric_weights = w[valid]
                metric_weights = metric_weights / metric_weights.sum()
                row[metric] = float((values[valid] * metric_weights).sum())

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    weights, isco_labels = load_employment_weights(EMPLOYMENT_PATH)
    ai_grouped, metric_columns = load_ai_by_group(AI_PATH)
    result = map_ai_to_isco(ai_grouped, metric_columns, weights, isco_labels)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, float_format="%.3f")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

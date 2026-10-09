from pathlib import Path
import re

import pandas as pd


INPUT_PATH = Path("../Data-science-mini-project/data/Unemployment by occupation by month.csv")
OUTPUT_PATH = Path("../Data-science-mini-project/data_processed/Unemployment by year ISCO 1-digit.csv")

UNEMPLOYMENT_SERIES = "Unemployed jobseekers on calculation date (number)"

# OC0 = Armed forces occupations
# OC6 = Skilled agricultural, forestry and fishery workers
ISCO_LABELS = {
    "OC1": "Managers",
    "OC2": "Professionals",
    "OC3": "Technicians and associate professionals",
    "OC4": "Clerical support workers",
    "OC5": "Service and sales workers",
    "OC7": "Craft and related trades workers",
    "OC8": "Plant and machine operators and assemblers",
    "OC9": "Elementary occupations",
}


def get_isco_group(column_name: str) -> str | None:
    """Return the ISCO 1-digit group for a 4-digit occupation column."""
    match = re.match(r"^(\d{4})\b", column_name)
    if match is None:
        return None

    major_group = match.group(1)[0]
    isco = f"OC{major_group}"

    return isco if isco in ISCO_LABELS else None


def main() -> None:
    # The Statistics Finland CSV is ISO-8859-1 encoded.
    df = pd.read_csv(
        INPUT_PATH,
        encoding="latin1",
        na_values=["..."],
    )

    # The file contains both unemployment and vacancy series. Keep only
    # unemployed jobseekers measured at the end-of-month calculation date.
    df = df[df["Information"] == UNEMPLOYMENT_SERIES].copy()

    if df.empty:
        raise ValueError(
            f"No rows found for information series: {UNEMPLOYMENT_SERIES!r}"
        )

    # Parse YYYY from values such as 2021M01.
    df["Year"] = pd.to_numeric(
        df["Month"].str.extract(r"^(\d{4})", expand=False),
        errors="coerce",
    ).astype("Int64")

    # Only produce annual values for complete calendar years.
    month_counts = df.groupby("Year")["Month"].nunique()
    complete_years = month_counts[month_counts == 12].index
    df = df[df["Year"].isin(complete_years)].copy()

    # Map the 4-digit occupation columns into ISCO major groups.
    columns_by_isco: dict[str, list[str]] = {
        isco: [] for isco in ISCO_LABELS
    }

    for column in df.columns:
        isco = get_isco_group(column)
        if isco is not None:
            columns_by_isco[isco].append(column)

    missing_groups = [
        isco for isco, columns in columns_by_isco.items() if not columns
    ]
    if missing_groups:
        raise ValueError(
            "No occupation columns found for ISCO groups: "
            + ", ".join(missing_groups)
        )

    monthly_rows = []

    for isco, occupation_columns in columns_by_isco.items():
        values = df[occupation_columns].apply(pd.to_numeric, errors="coerce")

        # Statistics Finland uses '...' for values subject to secrecy.
        # Those cells are omitted from the sum rather than treated as zero.
        monthly_total = values.sum(axis=1, min_count=1)

        group = pd.DataFrame(
            {
                "Year": df["Year"],
                "Month": df["Month"],
                "ISCO": isco,
                "Occupation": ISCO_LABELS[isco],
                "Unemployed jobseekers": monthly_total,
            }
        )
        monthly_rows.append(group)

    monthly = pd.concat(monthly_rows, ignore_index=True)

    # Convert monthly end-of-month counts to a yearly value by taking the
    # arithmetic mean of the 12 monthly observations.
    yearly = (
        monthly.groupby(["Year", "ISCO", "Occupation"], as_index=False)[
            "Unemployed jobseekers"
        ]
        .mean()
        .sort_values(["Year", "ISCO"])
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(OUTPUT_PATH, index=False, float_format="%.3f")

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

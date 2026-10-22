from pathlib import Path

import pandas as pd


INPUT_PATH = Path("../../data/Unemployment by occupation by month.csv")
OUTPUT_PATH = Path("../../data_processed/Unemployment by month ISCO 1-digit.csv")

UNEMPLOYMENT_SERIES = "Unemployed jobseekers on calculation date (number)"

# OC0 (armed forces) and OC6 (agriculture, forestry and fishery) are excluded.
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


def main() -> None:
    df = pd.read_csv(INPUT_PATH, encoding="latin1", na_values=["..."])
    df = df[df["Information"] == UNEMPLOYMENT_SERIES]
    if df.empty:
        raise ValueError(f"No rows found for {UNEMPLOYMENT_SERIES!r}")

    groups = []
    for isco, occupation in ISCO_LABELS.items():
        # Match four-digit occupation codes beginning with the group's digit.
        values = df.filter(regex=rf"^{isco[-1]}\d{{3}}\b")
        if values.empty:
            raise ValueError(f"No occupation columns found for {isco}")
        values = values.apply(pd.to_numeric, errors="coerce")

        groups.append(pd.DataFrame({
            "Month": df["Month"],
            "ISCO": isco,
            "Occupation": occupation,
            # As in the yearly script, omit suppressed/missing values.
            # A total with missing components is partial; all missing stays NaN.
            "Unemployed jobseekers": values.sum(axis=1, min_count=1),
        }))

    monthly = pd.concat(groups, ignore_index=True).sort_values(["Month", "ISCO"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_PATH, index=False, float_format="%.0f")
    print(f"Wrote {len(monthly)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

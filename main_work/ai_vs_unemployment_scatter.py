"""
Scatter plots: AI usage in enterprises vs. unemployment, by ISCO 1-digit group.

Left panel : every group-year point (AI % vs. unemployment indexed to 2021 = 100),
             one color per group, thin line connecting each group's years in order.
Right panel: change between two years per group (change in AI usage vs. % change in
             unemployment), with a trend line and correlation.

Requires: pandas, numpy, matplotlib
Put the two CSV files in the same folder as this script (or edit the paths below).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AI_FILE = "../Data-science-mini-project/data_processed/AI usage in enterprises by year ISCO 1-digit.csv"
UNEMP_FILE = "../Data-science-mini-project/data_processed/Unemployment by year ISCO 1-digit.csv"
AI_COL = "The enterprise uses Artificial Intelligence (AI) technologies, % of enterprises"
UNEMP_COL = "Unemployed jobseekers"

BASE_YEAR = 2021                    # unemployment index = 100 in this year (left panel)
START_YEAR, END_YEAR = 2023, 2025   # period for the change comparison (right panel)

ai = pd.read_csv(AI_FILE).rename(columns={AI_COL: "ai"})
un = pd.read_csv(UNEMP_FILE).rename(columns={UNEMP_COL: "unemp"})

# Keep only years that exist in both files (2022 has no AI data, so it drops out)
df = ai[["Year", "ISCO", "Occupation", "ai"]].merge(
    un[["Year", "ISCO", "unemp"]], on=["Year", "ISCO"]).dropna(subset=["ai"])

# Unemployment index relative to the base year, per group.
# Needed because groups differ in size, so raw counts can't be compared across groups.
base = un[un["Year"] == BASE_YEAR].set_index("ISCO")["unemp"]
df["unemp_idx"] = df["unemp"] / df["ISCO"].map(base) * 100

groups = sorted(df["ISCO"].unique())
names = df.drop_duplicates("ISCO").set_index("ISCO")["Occupation"]
colors = dict(zip(groups, plt.cm.tab10(np.linspace(0, 1, 10))[:len(groups)]))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 7))

# ---- Left: all group-year points ------------------------------------------
for g in groups:
    d = df[df["ISCO"] == g].sort_values("Year")
    axL.plot(d["ai"], d["unemp_idx"], color=colors[g], alpha=0.5, linewidth=1)
    axL.scatter(d["ai"], d["unemp_idx"], color=colors[g], s=55, zorder=3,
                label=f"{g}: {names[g]}")
    for _, r in d.iterrows():                     # label each point with its year
        axL.annotate(str(int(r["Year"]))[2:], (r["ai"], r["unemp_idx"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=7,
                     color=colors[g])
axL.axhline(100, color="grey", linestyle=":", linewidth=1)
axL.set_xlabel("Enterprises using AI (%)")
axL.set_ylabel(f"Unemployed jobseekers (index, {BASE_YEAR} = 100)")
axL.set_title("All group-year points (labels = year)")
axL.grid(alpha=0.3)
axL.legend(fontsize=8, loc="upper left")

# ---- Right: change between two years --------------------------------------
a0 = df[df["Year"] == START_YEAR].set_index("ISCO")
a1 = df[df["Year"] == END_YEAR].set_index("ISCO")
chg = pd.DataFrame({
    "d_ai": a1["ai"] - a0["ai"],                              # percentage points
    "d_un": (a1["unemp"] / a0["unemp"] - 1) * 100,            # percent
}).dropna()

for g, r in chg.iterrows():
    axR.scatter(r["d_ai"], r["d_un"], color=colors[g], s=110, zorder=3)
    axR.annotate(g, (r["d_ai"], r["d_un"]), xytext=(6, 6),
                 textcoords="offset points", fontsize=10)

r_val = np.corrcoef(chg["d_ai"], chg["d_un"])[0, 1]
slope, intercept = np.polyfit(chg["d_ai"], chg["d_un"], 1)
xs = np.linspace(chg["d_ai"].min(), chg["d_ai"].max(), 50)
axR.plot(xs, slope * xs + intercept, color="black", linestyle="--", linewidth=1,
         label=f"trend line (r = {r_val:.2f}, n = {len(chg)})")
axR.axhline(0, color="grey", linestyle=":", linewidth=1)
axR.set_xlabel(f"Change in AI usage {START_YEAR}→{END_YEAR} (percentage points)")
axR.set_ylabel(f"Change in unemployed jobseekers {START_YEAR}→{END_YEAR} (%)")
axR.set_title(f"Change per group, {START_YEAR}→{END_YEAR}")
axR.grid(alpha=0.3)
axR.legend(loc="best")

fig.suptitle("AI usage in enterprises vs. unemployment by ISCO occupation group", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("ai_vs_unemployment_scatter.png", dpi=150)

print(chg.round(1))
print("Right panel r =", round(r_val, 2))
print("Left panel pooled r =", round(np.corrcoef(df["ai"], df["unemp_idx"])[0, 1], 2))
plt.show()

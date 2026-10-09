"""
AI usage in enterprises vs. unemployment, one panel per ISCO 1-digit group.

Requires: pandas, matplotlib   (pip install pandas matplotlib)
Put the two CSV files in the same folder as this script (or edit the paths below).
"""
import textwrap
import pandas as pd
import matplotlib.pyplot as plt

AI_FILE = "../Data-science-mini-project/data_processed/AI usage in enterprises by year ISCO 1-digit.csv"
UNEMP_FILE = "../Data-science-mini-project/data_processed/Unemployment by year ISCO 1-digit.csv"

AI_COL = "The enterprise uses Artificial Intelligence (AI) technologies, % of enterprises"
UNEMP_COL = "Unemployed jobseekers"

AI_COLOR = "#1f77b4"     # blue
UNEMP_COLOR = "#d62728"  # red

ai = pd.read_csv(AI_FILE)
un = pd.read_csv(UNEMP_FILE)

# Group order and names (OC6 doesn't appear in your data)
groups = (un[["ISCO", "Occupation"]].drop_duplicates()
          .sort_values("ISCO").itertuples(index=False))
groups = list(groups)

years_un = range(int(un["Year"].min()), int(un["Year"].max()) + 1)
years_ai = range(int(ai["Year"].min()), int(ai["Year"].max()) + 1)

# One shared 0..max scale for AI so panels are comparable with each other.
# Unemployment gets its own scale per panel, because group sizes differ a lot
# (e.g. Managers ~2-3k vs Craft workers ~50k) and would flatten the small groups.
ai_max = ai[AI_COL].max() * 1.1

ncols = 4
nrows = -(-len(groups) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.6 * nrows), sharex=True)
axes = axes.flatten()

for ax1, (isco, name) in zip(axes, groups):
    u = (un[un["ISCO"] == isco].set_index("Year")[UNEMP_COL]
         .reindex(years_un))
    a = (ai[ai["ISCO"] == isco].set_index("Year")[AI_COL]
         .reindex(years_ai))

    # Unemployment (left axis, red)
    ax1.plot(u.index, u.values, color=UNEMP_COLOR, marker="o", markersize=3.5,
             linewidth=1.8)
    ax1.set_ylim(0, u.max() * 1.15)
    ax1.tick_params(axis="y", labelcolor=UNEMP_COLOR, labelsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax1.grid(alpha=0.3)

    # AI usage (right axis, blue)
    ax2 = ax1.twinx()
    valid = a.dropna()
    ax2.plot(valid.index, valid.values, color=AI_COLOR, linestyle="--",
             linewidth=1.2, alpha=0.6)                      # bridges missing 2022
    ax2.plot(a.index, a.values, color=AI_COLOR, marker="s", markersize=4,
             linewidth=1.8)
    ax2.set_ylim(0, ai_max)
    ax2.tick_params(axis="y", labelcolor=AI_COLOR, labelsize=8)

    ax1.set_title(f"{isco}: " + "\n".join(textwrap.wrap(name, 34)), fontsize=10)
    ax1.set_xticks(range(2009, 2026, 2))
    ax1.tick_params(axis="x", rotation=45, labelsize=8)

# Hide unused panels
for ax in axes[len(groups):]:
    ax.set_visible(False)

# Single legend for the whole figure
from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([0], [0], color=UNEMP_COLOR, marker="o", label="Unemployed jobseekers (left axis, own scale per panel)"),
    Line2D([0], [0], color=AI_COLOR, marker="s", label="Enterprises using AI, % (right axis, same scale in all panels)"),
], loc="lower center", ncol=2, frameon=False, fontsize=10)

fig.suptitle("AI usage in enterprises (2021–2025) vs. unemployment (2009–2025), by ISCO occupation group",
             fontsize=14)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig("ai_vs_unemployment_by_isco.png", dpi=150)
plt.show()

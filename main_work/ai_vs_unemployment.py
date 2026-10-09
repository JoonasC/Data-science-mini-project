"""
AI usage in enterprises vs. unemployment, overlaid on one chart.

Requires: pandas, matplotlib   (pip install pandas matplotlib)
Put the two CSV files in the same folder as this script (or edit the paths below).
"""
import pandas as pd
import matplotlib.pyplot as plt

AI_FILE = "../Data-science-mini-project/data_processed/AI usage in enterprises by year ISCO 1-digit.csv"
UNEMP_FILE = "../Data-science-mini-project/data_processed/Unemployment by year ISCO 1-digit.csv"

AI_COL = "The enterprise uses Artificial Intelligence (AI) technologies, % of enterprises"
UNEMP_COL = "Unemployed jobseekers"

AI_COLOR = "#1f77b4"     # blue
UNEMP_COLOR = "#d62728"  # red

# ---- Load -------------------------------------------------------------
ai = pd.read_csv(AI_FILE)
un = pd.read_csv(UNEMP_FILE)

# ---- Aggregate to "as a whole" per year -------------------------------
# AI usage is a percentage per occupation group, so it can't be summed:
# use the simple (unweighted) average across the occupation groups.
ai_year = (ai[ai["Year"] >= 2021]
           .groupby("Year")[AI_COL].mean()      # years with no data (2022) stay NaN
           .reindex(range(2021, int(ai["Year"].max()) + 1)))

# Unemployed jobseekers are counts, so total them across occupation groups.
un_year = (un[un["Year"] >= 2009]
           .groupby("Year")[UNEMP_COL].sum())

# ---- Plot: two y-axes sharing one x-axis ------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

# Unemployment (left axis)
ax1.plot(un_year.index, un_year.values, color=UNEMP_COLOR, marker="o",
         linewidth=2.2, label="Unemployed jobseekers (total)")
ax1.set_xlabel("Year")
ax1.set_ylabel("Unemployed jobseekers", color=UNEMP_COLOR)
ax1.tick_params(axis="y", labelcolor=UNEMP_COLOR)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax1.set_xticks(range(un_year.index.min(), un_year.index.max() + 1))
ax1.tick_params(axis="x", rotation=45)
ax1.grid(alpha=0.3)

# AI usage (right axis)
ax2 = ax1.twinx()
valid = ai_year.dropna()
# dashed line bridges the missing year (2022) so the gap is visible but the trend is readable
ax2.plot(valid.index, valid.values, color=AI_COLOR, linestyle="--",
         linewidth=1.5, alpha=0.6)
ax2.plot(ai_year.index, ai_year.values, color=AI_COLOR, marker="s",
         linewidth=2.2, label="AI usage in enterprises (avg. % across occupations)")
ax2.set_ylabel("Enterprises using AI (%)", color=AI_COLOR)
ax2.tick_params(axis="y", labelcolor=AI_COLOR)
ax2.set_ylim(0, None)

# One combined legend
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")

plt.title("AI usage in enterprises (2021–2025) vs. unemployment (2009–2025)")
fig.tight_layout()
fig.savefig("ai_vs_unemployment.png", dpi=150)
plt.show()

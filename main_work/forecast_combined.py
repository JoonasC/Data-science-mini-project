"""
Combined forecast: all ISCO groups together, AI usage and unemployment in one chart.

  Unemployment (all groups summed): damped-trend exponential smoothing (Holt) on the log
      scale, 80% prediction band from a residual bootstrap, plus a backtest vs. naive.
  AI usage (average % across groups): logistic growth curve fitted by nonlinear least
      squares, shown as scenarios for assumed ceilings.

Requires: pandas, numpy, scipy, matplotlib
Put the two CSV files in the same folder as this script (or edit the paths below).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit

AI_FILE = "../Data-science-mini-project/data_processed/AI usage in enterprises by year ISCO 1-digit.csv"
UNEMP_FILE = "../Data-science-mini-project/data_processed/Unemployment by year ISCO 1-digit.csv"
AI_COL = "The enterprise uses Artificial Intelligence (AI) technologies, % of enterprises"
UNEMP_COL = "Unemployed jobseekers"

HORIZON = 3
BACKTEST_FROM = 2022
CEILINGS = [70, 85, 100]
N_SIMS = 5000
RNG = np.random.default_rng(42)
UN_COLOR, AI_COLOR = "#d62728", "#1f77b4"

ai = pd.read_csv(AI_FILE).rename(columns={AI_COL: "ai"})
un = pd.read_csv(UNEMP_FILE).rename(columns={UNEMP_COL: "unemp"})

un_total = un.groupby("Year")["unemp"].sum().sort_index()          # counts -> sum
ai_mean = ai.dropna(subset=["ai"]).groupby("Year")["ai"].mean()    # percentages -> mean
last_year = int(un_total.index.max())
future = np.arange(last_year + 1, last_year + HORIZON + 1)


# ---- Holt damped-trend exponential smoothing --------------------------------
def _run(y, a, b, phi):
    lvl, trd = y[0], y[1] - y[0]
    errs = []
    for t in range(1, len(y)):
        pred = lvl + phi * trd
        errs.append(y[t] - pred)
        new = a * y[t] + (1 - a) * pred
        trd = b * (new - lvl) + (1 - b) * phi * trd
        lvl = new
    return np.array(errs), lvl, trd

def fit_holt(y):
    res = minimize(lambda p: np.sum(_run(y, *p)[0] ** 2), x0=[0.5, 0.3, 0.9],
                   bounds=[(0.01, 0.99), (0.01, 0.99), (0.8, 0.98)])
    a, b, phi = res.x
    errs, lvl, trd = _run(y, a, b, phi)
    return dict(a=a, b=b, phi=phi, lvl=lvl, trd=trd, errs=errs)

def point_forecast(m, h):
    return np.array([m["lvl"] + sum(m["phi"] ** i for i in range(1, k + 1)) * m["trd"]
                     for k in range(1, h + 1)])

def simulate(m, h, n=N_SIMS):
    out = np.empty((n, h))
    for s in range(n):
        lvl, trd = m["lvl"], m["trd"]
        for k in range(h):
            pred = lvl + m["phi"] * trd
            y = pred + RNG.choice(m["errs"])
            new = m["a"] * y + (1 - m["a"]) * pred
            trd = m["b"] * (new - lvl) + (1 - m["b"]) * m["phi"] * trd
            lvl = new
            out[s, k] = y
    return out

m = fit_holt(np.log(un_total.values))
un_fc = np.exp(point_forecast(m, HORIZON))
sims = np.exp(simulate(m, HORIZON))
un_lo, un_hi = np.percentile(sims, [10, 90], axis=0)

# backtest on the combined series
train = un_total[un_total.index <= BACKTEST_FROM]; test = un_total[un_total.index > BACKTEST_FROM]
mb = fit_holt(np.log(train.values))
bt_pred = np.exp(point_forecast(mb, len(test)))
bt = pd.DataFrame({"actual": test.values, "model": bt_pred,
                   "naive": train.values[-1]}, index=test.index)
print(f"BACKTEST (fit to {BACKTEST_FROM}, forecast {BACKTEST_FROM+1}-{last_year}):")
print(bt.round(0).to_string())
print(f"Mean abs % error  model: {np.mean(np.abs(bt.model / bt.actual - 1)) * 100:.1f}%   "
      f"naive: {np.mean(np.abs(bt.naive / bt.actual - 1)) * 100:.1f}%")

print("\nUNEMPLOYMENT FORECAST (all groups, 80% band):")
for y, f, lo, hi in zip(future, un_fc, un_lo, un_hi):
    print(f"  {y}: {f:,.0f}   [{lo:,.0f} - {hi:,.0f}]")


# ---- AI logistic scenarios ---------------------------------------------------
def logistic(t, r, t0, K):
    return K / (1 + np.exp(-r * (t - t0)))

grid = np.arange(int(ai_mean.index.min()), int(future[-1]) + 1)
ai_curves = {}
print("\nAI USAGE SCENARIOS (average across groups):")
for K in CEILINGS:
    (r, t0), _ = curve_fit(lambda t, r, t0: logistic(t, r, t0, K),
                           ai_mean.index.values, ai_mean.values,
                           p0=[0.5, 2025], bounds=([0.01, 2010], [3, 2045]))
    rmse = np.sqrt(np.mean((logistic(ai_mean.index.values, r, t0, K) - ai_mean.values) ** 2))
    ai_curves[K] = logistic(grid, r, t0, K)
    vals = ", ".join(f"{y}: {logistic(y, r, t0, K):.1f}%" for y in future)
    print(f"  ceiling {K}%: {vals}   (fit rmse {rmse:.1f} pts)")


# ---- One combined chart ------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(13, 6.5))
ax1.plot(un_total.index, un_total.values, color=UN_COLOR, marker="o", linewidth=2.2,
         label="Unemployed jobseekers, all groups (history)")
ax1.plot(np.r_[last_year, future], np.r_[un_total.values[-1], un_fc], color=UN_COLOR,
         linestyle="--", marker="o", linewidth=2, label="Unemployment forecast")
ax1.fill_between(future, un_lo, un_hi, color=UN_COLOR, alpha=0.2, label="80% prediction band")
ax1.set_xlabel("Year")
ax1.set_ylabel("Unemployed jobseekers", color=UN_COLOR)
ax1.tick_params(axis="y", labelcolor=UN_COLOR)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax1.set_xticks(range(2009, int(future[-1]) + 1))
ax1.tick_params(axis="x", rotation=45)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.scatter(ai_mean.index, ai_mean.values, color=AI_COLOR, marker="s", s=60, zorder=5,
            label="AI usage, average % (observed)")
shades = plt.cm.Blues(np.linspace(0.45, 0.9, len(CEILINGS)))
for (K, curve), c in zip(ai_curves.items(), shades):
    ax2.plot(grid, curve, color=c, linestyle="--", linewidth=1.6,
             label=f"AI scenario: ceiling {K}%")
ax2.set_ylabel("Enterprises using AI (%)", color=AI_COLOR)
ax2.tick_params(axis="y", labelcolor=AI_COLOR)
ax2.set_ylim(0, 100)

ax1.axvline(last_year + 0.5, color="grey", linestyle=":")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
plt.title(f"Combined forecast to {future[-1]}: unemployment (all groups) and AI usage "
          "(average across groups)")
fig.tight_layout()
fig.savefig("forecast_combined.png", dpi=150)
plt.show()

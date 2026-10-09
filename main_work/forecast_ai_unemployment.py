"""
Forecasting AI usage and unemployment by ISCO 1-digit group.

Statistical methods used
  1. Unemployment: damped-trend exponential smoothing (Holt's method, fitted on the
     log scale by least squares), with prediction bands from a residual bootstrap.
     Validated by a backtest against the naive "next year = this year" forecast.
  2. AI usage: logistic (S-curve) growth fitted by nonlinear least squares, shown as
     scenarios for an assumed saturation ceiling (4 data points can't estimate it).
  3. Link between the two: fixed-effects OLS regression of log(unemployment) on AI usage
     (with and without year effects), reported as a what-if with 95% confidence intervals.

Requires: pandas, numpy, scipy, matplotlib
Put the two CSV files in the same folder as this script (or edit the paths below).
"""
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize, curve_fit

AI_FILE = "../Data-science-mini-project/data_processed/AI usage in enterprises by year ISCO 1-digit.csv"
UNEMP_FILE = "../Data-science-mini-project/data_processed/Unemployment by year ISCO 1-digit.csv"
AI_COL = "The enterprise uses Artificial Intelligence (AI) technologies, % of enterprises"
UNEMP_COL = "Unemployed jobseekers"

HORIZON = 3                       # years to forecast beyond the last data year
BACKTEST_FROM = 2022              # fit up to this year, forecast the rest, compare with actuals
CEILINGS = [70, 85, 100]          # assumed maximum AI usage (%) for the scenarios
N_SIMS = 2000                     # bootstrap simulations for the prediction band
RNG = np.random.default_rng(42)
UN_COLOR, AI_COLOR = "#d62728", "#1f77b4"

ai = pd.read_csv(AI_FILE).rename(columns={AI_COL: "ai"})
un = pd.read_csv(UNEMP_FILE).rename(columns={UNEMP_COL: "unemp"})
groups = un[["ISCO", "Occupation"]].drop_duplicates().sort_values("ISCO")
last_year = int(un["Year"].max())
future = np.arange(last_year + 1, last_year + HORIZON + 1)


# =====================================================================
# 1. UNEMPLOYMENT: damped-trend exponential smoothing (Holt), log scale
# =====================================================================
def _run(y, a, b, phi):
    """Run the recursion; return one-step-ahead errors and final (level, trend)."""
    lvl, trd = y[0], y[1] - y[0]
    errs = []
    for t in range(1, len(y)):
        pred = lvl + phi * trd
        errs.append(y[t] - pred)
        new_lvl = a * y[t] + (1 - a) * pred
        trd = b * (new_lvl - lvl) + (1 - b) * phi * trd
        lvl = new_lvl
    return np.array(errs), lvl, trd

def fit_holt(y):
    sse = lambda p: np.sum(_run(y, *p)[0] ** 2)
    res = minimize(sse, x0=[0.5, 0.3, 0.9], bounds=[(0.01, 0.99), (0.01, 0.99), (0.8, 0.98)])
    a, b, phi = res.x
    errs, lvl, trd = _run(y, a, b, phi)
    return dict(a=a, b=b, phi=phi, lvl=lvl, trd=trd, errs=errs)

def point_forecast(m, h):
    return np.array([m["lvl"] + sum(m["phi"] ** i for i in range(1, k + 1)) * m["trd"]
                     for k in range(1, h + 1)])

def simulate(m, h, n=N_SIMS):
    """Future paths: step the model forward, adding bootstrapped past errors."""
    out = np.empty((n, h))
    for s in range(n):
        lvl, trd = m["lvl"], m["trd"]
        for k in range(h):
            pred = lvl + m["phi"] * trd
            y = pred + RNG.choice(m["errs"])
            new_lvl = m["a"] * y + (1 - m["a"]) * pred
            trd = m["b"] * (new_lvl - lvl) + (1 - m["b"]) * m["phi"] * trd
            lvl = new_lvl
            out[s, k] = y
    return out

un_fc, backtest = {}, []
for isco, name in groups.itertuples(index=False):
    s = un[un["ISCO"] == isco].set_index("Year")["unemp"].sort_index()
    ly = np.log(s.values)
    m = fit_holt(ly)
    sims = np.exp(simulate(m, HORIZON))
    un_fc[isco] = dict(mean=np.exp(point_forecast(m, HORIZON)),
                       lo=np.percentile(sims, 10, axis=0), hi=np.percentile(sims, 90, axis=0))
    # backtest: fit on data up to BACKTEST_FROM, forecast the rest
    train = s[s.index <= BACKTEST_FROM]; test = s[s.index > BACKTEST_FROM]
    mb = fit_holt(np.log(train.values))
    pred = np.exp(point_forecast(mb, len(test)))
    naive = np.repeat(train.values[-1], len(test))
    backtest.append(dict(ISCO=isco,
                         model_err_pct=np.mean(np.abs(pred / test.values - 1)) * 100,
                         naive_err_pct=np.mean(np.abs(naive / test.values - 1)) * 100))

bt = pd.DataFrame(backtest).set_index("ISCO")
print(f"\nBACKTEST: fit to {BACKTEST_FROM}, forecast {BACKTEST_FROM+1}-{last_year} "
      f"(mean absolute % error)")
print(bt.round(1).to_string())
print(f"Average: model {bt.model_err_pct.mean():.1f}%  vs  naive {bt.naive_err_pct.mean():.1f}%")
print("Model beats naive in", int((bt.model_err_pct < bt.naive_err_pct).sum()), "of", len(bt), "groups")

fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
for ax, (isco, name) in zip(axes.flatten(), groups.itertuples(index=False)):
    s = un[un["ISCO"] == isco].set_index("Year")["unemp"].sort_index()
    f = un_fc[isco]
    ax.plot(s.index, s.values, color=UN_COLOR, marker="o", markersize=3.5, linewidth=1.8)
    ax.plot(np.r_[last_year, future], np.r_[s.values[-1], f["mean"]], color=UN_COLOR,
            linestyle="--", marker="o", markersize=3.5)
    ax.fill_between(future, f["lo"], f["hi"], color=UN_COLOR, alpha=0.2)
    ax.set_ylim(0, max(s.max(), f["hi"].max()) * 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(f"{isco}: " + "\n".join(textwrap.wrap(name, 34)), fontsize=10)
    ax.set_xticks(range(2009, int(future[-1]) + 1, 3))
    ax.tick_params(axis="x", rotation=45, labelsize=8); ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.3)
fig.suptitle(f"Unemployed jobseekers: damped-trend exponential smoothing forecast to "
             f"{future[-1]} (shaded = 80% prediction band)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("forecast_unemployment.png", dpi=150)


# =====================================================================
# 2. AI USAGE: logistic growth curve, nonlinear least squares, scenarios
# =====================================================================
def logistic(t, r, t0, K):
    return K / (1 + np.exp(-r * (t - t0)))

grid = np.arange(2021, last_year + HORIZON + 1)
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(CEILINGS)))
fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
fit_rows = []
for ax, (isco, name) in zip(axes.flatten(), groups.itertuples(index=False)):
    d = ai[ai["ISCO"] == isco].dropna(subset=["ai"]).sort_values("Year")
    ax.scatter(d["Year"], d["ai"], color=AI_COLOR, zorder=5, s=40, label="observed")
    for K, c in zip(CEILINGS, cmap):
        if K <= d["ai"].max():
            continue
        try:
            (r, t0), _ = curve_fit(lambda t, r, t0: logistic(t, r, t0, K),
                                   d["Year"].values, d["ai"].values,
                                   p0=[0.5, 2025], bounds=([0.01, 2010], [3, 2045]))
        except RuntimeError:
            continue
        rmse = np.sqrt(np.mean((logistic(d["Year"].values, r, t0, K) - d["ai"].values) ** 2))
        fit_rows.append(dict(ISCO=isco, ceiling=K, rate=r, midpoint_year=t0, rmse=rmse,
                             **{f"AI% {y}": logistic(y, r, t0, K) for y in future}))
        ax.plot(grid, logistic(grid, r, t0, K), color=c, label=f"ceiling {K}%")
    ax.axvline(last_year + 0.5, color="grey", linestyle=":")
    ax.set_title(f"{isco}: " + "\n".join(textwrap.wrap(name, 34)), fontsize=10)
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    ax.set_xticks(range(2021, int(grid[-1]) + 1))
axes[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle("AI usage in enterprises (%): logistic-curve scenarios "
             "(assumptions, not predictions)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("forecast_ai_scenarios.png", dpi=150)

fits = pd.DataFrame(fit_rows)
print("\nAI LOGISTIC FITS (rmse = misfit in percentage points on the observed points)")
print(fits.round(1).to_string(index=False))


# =====================================================================
# 3. WHAT-IF: fixed-effects OLS, log(unemployment) ~ AI usage
# =====================================================================
df = ai[["Year", "ISCO", "ai"]].merge(un[["Year", "ISCO", "unemp"]], on=["Year", "ISCO"]).dropna()
df["y"] = np.log(df["unemp"])

def ols(df, year_effects):
    X = [df["ai"].values]
    names = ["ai"]
    for g in sorted(df["ISCO"].unique()):
        X.append((df["ISCO"] == g).astype(float).values)
    if year_effects:
        for yr in sorted(df["Year"].unique())[1:]:
            X.append((df["Year"] == yr).astype(float).values)
    X = np.column_stack(X); y = df["y"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - np.linalg.matrix_rank(X)
    cov = np.linalg.pinv(X.T @ X) * (resid @ resid / dof)
    se = np.sqrt(cov[0, 0]); tcrit = stats.t.ppf(0.975, dof)
    return beta[0], se, tcrit, dof

print("\nWHAT-IF: change in unemployment associated with +10 percentage points of AI usage")
print("(fixed-effects OLS on log(unemployment); association only, not causation)")
for label, ye in [("group effects only", False), ("group + year effects", True)]:
    b, se, tc, dof = ols(df, ye)
    f = lambda x: (np.exp(10 * x) - 1) * 100
    print(f"  {label:22s}: {f(b):+6.1f}%   95% CI [{f(b - tc*se):+.1f}%, {f(b + tc*se):+.1f}%]   "
          f"(n={len(df)}, df={dof})")
print("  'group effects only' mixes AI with the common time trend; "
      "'group + year effects' uses only differences between groups.")

plt.show()

"""Backtests that lie: one momentum strategy, four backtests, watch the Sharpe ratio fall.

Strategy: time-series momentum on SPY. Long if the trailing L-day return is positive, short if negative.

  1. Naive        : signal uses today's close to trade today's return (look-ahead),
                    best lookback L picked on the whole test period, no costs
  2. No look-ahead: trade today on yesterday's signal
  3. + costs      : pay COST_BPS per unit of turnover (a full long->short flip is 2 units)
  4. Walk-forward : each year, pick L using only the previous TRAIN_YEARS, then trade that year out of sample

Run: python backtest.py   -> prints the table, writes sharpe_decay.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOOKBACKS = [5, 10, 20, 60, 120, 250]
COST_BPS = 5          # per unit turnover: ~1bp spread + slippage + fees for a liquid ETF, on the generous side of real
TRAIN_YEARS = 3
DATA = Path(__file__).with_name("spy.csv")


def load_prices():
    if not DATA.exists():  # cached so results are reproducible and the script runs offline
        import yfinance as yf
        px = yf.download("SPY", start="2000-01-01", end="2026-11-01", auto_adjust=True, progress=False)["Close"]
        px.squeeze().rename("close").to_csv(DATA)
    return pd.read_csv(DATA, index_col=0, parse_dates=True)["close"]


def strategy_returns(prices, lookback, lookahead=False, cost_bps=0.0):
    rets = prices.pct_change()
    position = np.sign(prices / prices.shift(lookback) - 1)  # uses prices up to and including today
    held = position if lookahead else position.shift(1)        # the bug vs the fix
    turnover = held.diff().abs()
    return (held * rets - turnover * cost_bps / 1e4).dropna()


def sharpe(r):
    return np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else 0.0


def best_in_sample(prices, period, **kw):
    """Pick the lookback with the best Sharpe *on the period we then report*: selection bias built in."""
    runs = {L: strategy_returns(prices, L, **kw).loc[period[0]:period[1]] for L in LOOKBACKS}
    L = max(runs, key=lambda L: sharpe(runs[L]))
    return L, runs[L]


def walk_forward(prices, cost_bps):
    runs = {L: strategy_returns(prices, L, cost_bps=cost_bps) for L in LOOKBACKS}
    years = sorted(set(prices.index.year))
    out, picks = [], {}
    for y in years[TRAIN_YEARS + 1:]:  # +1: the first year is eaten by the longest lookback
        train = slice(f"{y - TRAIN_YEARS}-01-01", f"{y - 1}-12-31")
        L = max(runs, key=lambda L: sharpe(runs[L].loc[train]))
        picks[y] = L
        out.append(runs[L].loc[str(y)])
    return pd.concat(out), picks


def run(prices):
    wf, picks = walk_forward(prices, COST_BPS)
    period = (wf.index[0], wf.index[-1])  # every row is scored on the same years
    rows = []
    for name, kw in [("1. Naive (look-ahead, best L, no costs)", dict(lookahead=True)),
                     ("2. Look-ahead removed", dict()),
                     (f"3. + {COST_BPS}bp transaction costs", dict(cost_bps=COST_BPS))]:
        L, r = best_in_sample(prices, period, **kw)
        rows.append((name, f"L={L}", r))
    rows.append(("4. Walk-forward (out of sample)", "L re-picked yearly", wf))
    rows.append(("Buy and hold SPY", "", prices.pct_change().loc[period[0]:period[1]]))
    return rows, picks


if __name__ == "__main__":
    prices = load_prices()
    rows, picks = run(prices)
    print(f"SPY {rows[-1][2].index[0]:%Y-%m-%d} to {rows[-1][2].index[-1]:%Y-%m-%d}\n")
    print(f"{'Backtest':42}{'Choice':20}{'Sharpe':>8}{'Ann. return':>13}")
    for name, choice, r in rows:
        print(f"{name:42}{choice:20}{sharpe(r):8.2f}{r.mean() * 252:13.1%}")
    print("\nWalk-forward lookback picks:", picks)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})
    names = [n.split(" (")[0] for n, _, _ in rows]
    colors = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#7f7f7f"]
    bars = a1.barh(names[::-1], [sharpe(r) for *_, r in rows][::-1], color=colors[::-1])
    a1.bar_label(bars, fmt="%.2f", padding=3)
    a1.axvline(0, color="black", lw=0.8)
    a1.set_xlim(right=max(sharpe(r) for *_, r in rows) * 1.15)
    a1.set(title="Same strategy, four backtests: Sharpe ratio", xlabel="Annualized Sharpe")
    for (name, _, r), c in zip(rows, colors):
        a2.plot((1 + r).cumprod(), color=c, lw=1.2, label=name.split(" (")[0])
    a2.set_yscale("log")
    a2.set(title="Growth of $1 (log scale)")
    a2.legend(frameon=False, fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("sharpe_decay.png", dpi=150)

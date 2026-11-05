"""The point of the project, as a test: on pure noise, the naive backtest still finds a 'strategy'.
Run: python -m pytest -q"""
import numpy as np
import pandas as pd

from backtest import sharpe, strategy_returns, walk_forward


def random_walk(seed=0, days=252 * 20):
    rets = np.random.default_rng(seed).normal(0.0003, 0.01, days)  # no momentum by construction
    idx = pd.bdate_range("2000-01-03", periods=days)
    return pd.Series(100 * np.cumprod(1 + rets), index=idx)


def test_look_ahead_invents_an_edge_on_noise():
    prices = random_walk()
    assert sharpe(strategy_returns(prices, 5, lookahead=True)) > 2       # "amazing strategy"
    assert abs(sharpe(strategy_returns(prices, 5))) < 0.6                # honest: nothing there


def test_costs_only_ever_hurt():
    prices = random_walk(1)
    free, costly = strategy_returns(prices, 10), strategy_returns(prices, 10, cost_bps=5)
    assert (costly <= free + 1e-15).all() and costly.sum() < free.sum()


def test_walk_forward_never_sees_the_future():
    # Change only the final year's prices: earlier out-of-sample picks must not move.
    prices = random_walk(2)
    _, picks_a = walk_forward(prices, cost_bps=5)
    tampered = prices.copy()
    last_year = tampered.index.year == tampered.index.year.max()
    tampered[last_year] *= np.linspace(1, 3, last_year.sum())
    _, picks_b = walk_forward(tampered, cost_bps=5)
    final = max(picks_a)
    assert {y: L for y, L in picks_a.items() if y != final} == {y: L for y, L in picks_b.items() if y != final}

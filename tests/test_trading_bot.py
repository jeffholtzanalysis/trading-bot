import csv
import sys
import types

import numpy as np
import pandas as pd

import trading_bot as tb


def fake_prices(n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n)), index=idx)


def test_signal_is_zero_until_slow_average_exists():
    s = tb.signal(fake_prices())
    assert set(s.unique()) <= {0, 1}
    assert (s.iloc[: tb.SLOW - 1] == 0).all()


def test_signal_owns_in_steady_uptrend():
    close = pd.Series(np.linspace(100, 200, 120))
    assert tb.signal(close).iloc[-1] == 1


def test_backtest_buy_hold_matches_price_change():
    close = fake_prices()
    r = tb.backtest(close)
    expected = close.iloc[-1] / close.iloc[0] - 1
    assert abs(r["buy_hold"][0] - expected) < 1e-9
    assert r["trades"] >= 0


def test_backtest_does_not_peek_ahead():
    # Strategy can't earn the day-one return because it acts the day after a signal.
    close = pd.Series(np.linspace(100, 200, 120))
    r = tb.backtest(close)
    assert r["strategy"][0] < r["buy_hold"][0]


def test_extract_json_object():
    assert tb.extract_json_object('Sure! {"score": 0.5, "reason": "ok"} done')["score"] == 0.5


def test_news_sentiment_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    score, reason = tb.news_sentiment("SPY")
    assert score is None and "ANTHROPIC_API_KEY" in reason


def test_news_sentiment_parses_and_clamps(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(tb, "get_headlines", lambda s: ["- Big news"])
    reply = types.SimpleNamespace(
        content=[types.SimpleNamespace(text='{"score": 3, "reason": "very upbeat"}')]
    )
    fake = types.SimpleNamespace(
        Anthropic=lambda: types.SimpleNamespace(
            messages=types.SimpleNamespace(create=lambda **k: reply)
        )
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    assert tb.news_sentiment("SPY") == (1.0, "very upbeat")


def test_log_row_keeps_fixed_header(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "LOG_FILE", str(tmp_path / "log.csv"))
    tb.log_row({"date": "d1", "symbol": "SPY", "action": "hold"})
    tb.log_row({"symbol": "SPY", "date": "d2", "extra": "ignored"})
    with open(tb.LOG_FILE) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == tb.LOG_FIELDS
    assert [r["date"] for r in rows] == ["d1", "d2"]

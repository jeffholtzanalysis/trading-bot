"""
Starter trading bot: moving-average crossover, backtest + Alpaca PAPER trading.

Strategy: own the stock when the 20-day average is above the 50-day average,
sit in cash otherwise. It's simple on purpose; the point is to learn the plumbing.

SETUP (one time)
  1. Install Python 3.10+.
  2. pip install alpaca-py yfinance pandas matplotlib anthropic
  3. Make a free account at alpaca.markets, switch to Paper Trading,
     and generate API keys.
  4. Put your keys in a file named .env next to this script (copy .env.example),
     or set them as environment variables. Never paste them into this file:
       Mac/Linux:  export ALPACA_API_KEY="..."  and  export ALPACA_SECRET_KEY="..."
       Windows:    setx ALPACA_API_KEY "..."     and  setx ALPACA_SECRET_KEY "..."
  5. For the AI news step, get an API key at console.anthropic.com and set it the same way:
       ANTHROPIC_API_KEY="..."   (optional: without it the bot still trades, just logs no sentiment)

USAGE
  python trading_bot.py backtest            # test on 5 years of history, saves a chart
  python trading_bot.py backtest AAPL       # different symbol
  python trading_bot.py backtest SPY --period 10y
  python trading_bot.py trade               # check signal once and trade (paper)
  python trading_bot.py news                # just show Claude's read on today's news

  Every "trade" run adds a row to trade_log.csv: date, price, signal, action,
  and Claude's news sentiment (-1 very negative ... +1 very positive) with a one-line reason.
  The sentiment is LOGGED, not used to trade. After a few months, compare it to what
  the price did next. If it never helps, you've learned something real.

  Run "trade" once a day, e.g. around 3:30 PM Central. Daily strategy = daily check.

SAFETY
  - paper=True is hard-coded below. Don't change it until you've paper traded for months.
  - Create a file named STOP in the same folder and the bot refuses to trade (kill switch).
  - The bot never puts more than MAX_POSITION_PCT of your account into one trade.
  - If the account is down more than DAILY_LOSS_LIMIT_PCT today, it does nothing.

Not financial advice. This is a learning project.
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

try:  # optional: load keys from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---- Settings you can tweak ----
SYMBOL = "SPY"
FAST = 20                    # short moving average (days)
SLOW = 50                    # long moving average (days)
FEE_PCT = 0.0005             # 0.05% per trade, for slippage/spread
MAX_POSITION_PCT = 0.10      # use at most 10% of the account
DAILY_LOSS_LIMIT_PCT = 0.02  # stop for the day if account is down 2%
PAPER = True                 # keep this True
AI_MODEL = "claude-haiku-5-5"  # small, cheap model; fine for reading headlines
LOG_FILE = "trade_log.csv"
LOG_FIELDS = ["date", "symbol", "price", "signal", "action", "sentiment", "reason", "equity"]


# ---------- Data ----------
def get_prices(symbol: str, period: str = "5y") -> pd.Series:
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"No data for {symbol}. Check the ticker.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns multi-column
        close = close.iloc[:, 0]
    return close.dropna()


def signal(close: pd.Series) -> pd.Series:
    """1 = want to own it, 0 = want to be in cash."""
    fast = close.rolling(FAST).mean()
    slow = close.rolling(SLOW).mean()
    return (fast > slow).astype(int)


# ---------- Backtest ----------
def backtest(close: pd.Series) -> dict:
    sig = signal(close)
    position = sig.shift(1).fillna(0)          # act the day AFTER the signal (no peeking)
    daily = close.pct_change().fillna(0)
    trades = position.diff().abs().fillna(0)
    strat = position * daily - trades * FEE_PCT

    strat_curve = (1 + strat).cumprod()
    hold_curve = (1 + daily).cumprod()
    years = len(close) / 252

    def stats(curve):
        total = curve.iloc[-1] - 1
        cagr = curve.iloc[-1] ** (1 / years) - 1
        max_dd = (curve / curve.cummax() - 1).min()
        return total, cagr, max_dd

    return {
        "strategy": stats(strat_curve),
        "buy_hold": stats(hold_curve),
        "trades": int(trades.sum()),
        "strat_curve": strat_curve,
        "hold_curve": hold_curve,
    }


def run_backtest(symbol: str, period: str = "5y"):
    close = get_prices(symbol, period)
    r = backtest(close)
    print(f"\nBacktest: {symbol}, {close.index[0].date()} to {close.index[-1].date()}")
    print(f"{'':12}{'Total':>10}{'Per year':>10}{'Worst drop':>12}")
    for name, key in [("Strategy", "strategy"), ("Buy & hold", "buy_hold")]:
        t, c, d = r[key]
        print(f"{name:12}{t:>10.1%}{c:>10.1%}{d:>12.1%}")
    print(f"Trades: {r['trades']}")
    if r["strategy"][0] < r["buy_hold"][0]:
        print("-> Strategy LOST to just holding. That's common, and it's the honest answer.")

    try:
        import matplotlib

        matplotlib.use("Agg")  # save to file, no window needed
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(r["strat_curve"], label=f"MA {FAST}/{SLOW} strategy")
        plt.plot(r["hold_curve"], label="Buy & hold")
        plt.title(f"{symbol}: growth of $1")
        plt.legend()
        plt.grid(alpha=0.3)
        out = f"backtest_{symbol}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"Chart saved: {out}")
    except ImportError:
        print("(pip install matplotlib to get a chart)")


# ---------- AI news sentiment ----------
def get_headlines(symbol: str, limit: int = 15) -> list[str]:
    """Last ~2 days of headlines from Alpaca's free news feed."""
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
    start = datetime.now(timezone.utc) - timedelta(days=2)
    result = client.get_news(NewsRequest(symbols=symbol, start=start, limit=limit))
    items = getattr(result, "news", None) or result.data.get("news", [])
    return [f"- {n.headline}" + (f": {n.summary[:200]}" if n.summary else "") for n in items]


def extract_json_object(text: str) -> dict:
    """Pull the first {...} block out of the AI reply and parse it."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("AI reply had no JSON")
    return json.loads(m.group(0))


def news_sentiment(symbol: str) -> tuple[float | None, str]:
    """Ask Claude to score the news. Returns (score from -1 to 1, one-line reason)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None, "no ANTHROPIC_API_KEY set"
    try:
        headlines = get_headlines(symbol)
    except Exception as e:
        return None, f"news fetch failed: {e}"
    if not headlines:
        return 0.0, "no news in the last 2 days"

    import anthropic
    prompt = (
        f"Here are recent news headlines about {symbol}:\n\n" + "\n".join(headlines) +
        "\n\nRate the overall sentiment for the stock's short-term outlook from -1 "
        "(very negative) to 1 (very positive), 0 if neutral or mixed. Don't predict the price; "
        "just judge the tone of the news. Reply with ONLY JSON like "
        '{"score": 0.2, "reason": "one short sentence"}'
    )
    try:
        resp = anthropic.Anthropic().messages.create(
            model=AI_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(getattr(part, "text", "") for part in resp.content)
        data = extract_json_object(text)
        score = max(-1.0, min(1.0, float(data["score"])))
        return score, str(data.get("reason", ""))[:200]
    except Exception as e:
        return None, f"AI step failed: {e}"


def log_row(row: dict):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def run_news(symbol: str):
    score, reason = news_sentiment(symbol)
    print(f"{symbol} news sentiment: {score if score is not None else 'n/a'} | {reason}")


# ---------- Paper trading ----------
def run_trade(symbol: str):
    if os.path.exists("STOP"):
        print("STOP file found. Not trading.")
        return

    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY first (see top of file).")

    client = TradingClient(key, secret, paper=PAPER)
    acct = client.get_account()
    equity, last_equity = float(acct.equity), float(acct.last_equity)

    if last_equity > 0 and (equity - last_equity) / last_equity < -DAILY_LOSS_LIMIT_PCT:
        print(f"Account down more than {DAILY_LOSS_LIMIT_PCT:.0%} today. Standing down.")
        return

    close = get_prices(symbol, period="1y")
    want = int(signal(close).iloc[-1])
    price = float(close.iloc[-1])

    try:
        qty_held = float(client.get_open_position(symbol).qty)
    except Exception:
        qty_held = 0.0

    side = "OWN" if want else "CASH"
    print(f"{symbol} ${price:.2f} | signal: {side} | holding {qty_held:g} shares")

    score, reason = news_sentiment(symbol)
    print(f"News sentiment: {score if score is not None else 'n/a'} | {reason}")

    if want and qty_held == 0:
        qty = int(equity * MAX_POSITION_PCT // price)
        if qty < 1:
            action = "none (position limit < 1 share)"
        else:
            try:
                client.submit_order(MarketOrderRequest(
                    symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
                action = f"BUY {qty}"
            except Exception as e:
                action = f"BUY failed: {e}"
    elif not want and qty_held > 0:
        try:
            client.close_position(symbol)
            action = f"SELL {qty_held:g}"
        except Exception as e:
            action = f"SELL failed: {e}"
    else:
        action = "hold"
    print(f"Action: {action}")

    log_row({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "symbol": symbol,
        "price": round(price, 2),
        "signal": "OWN" if want else "CASH",
        "action": action,
        "sentiment": "" if score is None else round(score, 2),
        "reason": reason,
        "equity": round(equity, 2),
    })


def main(argv=None):
    p = argparse.ArgumentParser(description="Moving-average crossover bot (paper trading).")
    p.add_argument("mode", nargs="?", choices=["backtest", "trade", "news"], default="backtest")
    p.add_argument("symbol", nargs="?", default=SYMBOL)
    p.add_argument("--period", default="5y", help="backtest history, e.g. 1y, 5y, 10y, max")
    args = p.parse_args(argv)
    sym = args.symbol.upper()
    if args.mode == "backtest":
        run_backtest(sym, args.period)
    elif args.mode == "trade":
        run_trade(sym)
    else:
        run_news(sym)


if __name__ == "__main__":
    main()

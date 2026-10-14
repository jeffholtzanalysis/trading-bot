# Trading Bot: Moving-Average Crossover + AI News Sentiment

A learning project that backtests a simple trading strategy, runs it daily in an Alpaca **paper** (fake-money) account, and has Claude score the day's news so the results can be analyzed later.

**Strategy:** own the stock when its 20-day average price is above its 50-day average, and hold cash otherwise.

**The question this project answers:** does a simple rule-based strategy beat buying and holding? And does AI-scored news sentiment tell us anything about what the price does next? The honest answer may be "no." Finding that out with real data is the point.

> Educational project. Paper trading only. Not financial advice.

## What it does

| Command | What happens |
|---|---|
| `python trading_bot.py backtest` | Tests the strategy on 5 years of SPY, compares it to buy-and-hold, saves a chart |
| `python trading_bot.py backtest AAPL --period 10y` | Same, for another stock or time range |
| `python trading_bot.py news` | Shows Claude's sentiment score for today's headlines |
| `python trading_bot.py trade` | Checks today's signal, places a paper trade if needed, logs everything to `trade_log.csv` |

Each `trade` run writes one row to `trade_log.csv` with the date, price, signal, action, Claude's sentiment score (-1 to +1), a one-line reason, and the account value. **Sentiment is logged only. It never changes a trade.**

## Setup

```bash
git clone https://github.com/jeffholtzanalysis/trading-bot.git
cd trading-bot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python trading_bot.py backtest     # works without any keys
```

For trading and news, copy `.env.example` to `.env` and add:
- **Alpaca paper keys:** free account at [alpaca.markets](https://alpaca.markets), switch to Paper Trading, generate keys
- **Anthropic key (optional):** [console.anthropic.com](https://console.anthropic.com). Without it the bot still trades and simply logs no sentiment.

`.env` is in `.gitignore`, so your keys never get uploaded.

## Run it automatically (optional)

The workflow `.github/workflows/paper-trade.yml` runs `trade` every weekday during market hours on GitHub's servers and commits the updated `trade_log.csv` back to this repo. It does nothing until you add repo secrets under **Settings → Secrets and variables → Actions**:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ANTHROPIC_API_KEY` (optional)

To pause it, open the **Actions** tab, select "Daily paper trade," and choose **Disable workflow**.

## Safety

- `PAPER = True` is hard-coded. Don't change it.
- A file named `STOP` in the folder makes the bot refuse to trade (kill switch).
- No more than 10% of the account goes into a position.
- If the account is down 2% on the day, the bot does nothing.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

Tests run automatically on every push (`.github/workflows/ci.yml`).

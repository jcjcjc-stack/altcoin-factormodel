# Altcoin Factor Model

This project researches whether altcoin hourly returns can be explained or forecast using crypto market factors, market proxy ETFs, macro liquidity/rate data, BTC funding, and BTC volatility data.

The main workflow is:

1. Configure the research window and symbols in `refresh_data.py`.
2. Download the latest local research datasets.
3. Run the notebooks to compare factor relationships, forecast quality, and trading performance.

Raw downloaded data is not committed to git. It is rebuilt locally from the downloader scripts.

## Project Layout

```text
research_notebooks/
  PEPE/
    PEPE_Research_Regression.ipynb
    PEPE_Trading_Forecast_Regression.ipynb
  SOL/
    SOL_Research_Regression.ipynb
    SOL_Trading_Forecast_Regression.ipynb
src/
  paths.py
  load_data.py
  features.py
  modeling.py
  backtest.py
  plots.py
data/
  raw/
  data_inventory.csv
```

## Notebooks

Each coin folder has two notebooks with the same workflow.

### Research Regression

Same-hour explanatory regression.

Use this notebook to answer:

- Which factors move with the target coin in the same hour?
- How much of the target coin's hourly return variance is explained by crypto, macro, ETF proxy, funding, and DVOL features?
- Which features survive Ridge and Elastic Net regularization?
- Are the regression diagnostics reasonable?

This notebook is mainly for factor research and interpretation. It is not the trading backtest.

### Trading Forecast Regression

Out-of-sample lagged forecast and trading notebook.

Use this notebook to answer:

- Can lagged features forecast the target coin's next hourly return?
- How do OLS, Ridge, and Elastic Net compare out of sample?
- Which Elastic Net features are actually used?
- Do forecast signals produce profitable trades on the test period?
- What are the strategy metrics: cumulative return, CAGR, Sharpe, max drawdown, variance, Sortino, directional accuracy, and trade count?

The trading notebook uses a time split: the first 80% of rows are used for training, and the final 20% are used as the out-of-sample test period.

## Data Pipeline

The active data sources are:

- Binance Spot OHLCV: `PEPEUSDT`, `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `SHIBUSDT`
- Yahoo Finance hourly market proxies: `QQQ`, `SPY`, `IEF`, `TLT`, `UUP`, `HYG`, `VXX`
- FRED macro data: `M2 money supply`, `SOFR`, `effective fed funds rate`
- Binance Futures funding data: `BTCUSDT`
- Deribit DVOL: `BTC implied volatility candles`

CryptoCompare has been removed from the active pipeline.

## Setup

Create or activate a Python environment, then install the research dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels yfinance requests notebook
```

If you already use Anaconda, install these packages into that environment and open the notebooks from there.

## Configure The Research Run

Edit the top section of `refresh_data.py`.

Key settings:

```python
RESEARCH_START_DATE = "2024-08-02"
RESEARCH_START_TIME = "2024-08-02 00:00:00+00:00"
RESEARCH_TIME_INTERVAL = "1H"
```

Supported intervals:

- `1H`: hourly
- `1D`: daily
- `1M`: monthly
- `1Y`: yearly

Hourly mode is the main setup. Yahoo Finance hourly data has a limited lookback window, so the start date is set to stay inside that limit.

You can also change the symbols and tickers in `refresh_data.py`:

- `BINANCE_SPOT_SYMBOLS`
- `YFINANCE_TICKERS`
- `FRED_SERIES`
- `BINANCE_BTC_FUNDING_SYMBOL`
- `DERIBIT_DVOL_CURRENCY`

## Refresh Data

Preview the current config:

```bash
python refresh_data.py --dry-run
```

Download all active datasets:

```bash
python refresh_data.py
```

This writes local CSVs under `data/raw/` and rebuilds `data/data_inventory.csv`.

Main files used by the notebooks:

- `data/raw/binance_daily/binance_spot_1H_ohlcv_combined.csv`
- `data/raw/yfinance_market/yfinance_market_1H_ohlcv_combined.csv`
- `data/raw/fred_macro/fred_macro_1D_combined.csv`
- `data/raw/binance_funding/BTCUSDT_funding_rate_1H.csv`
- `data/raw/deribit_dvol/BTC_DVOL_1H.csv`

## Run The Analysis

After refreshing data, open Jupyter:

```bash
jupyter notebook
```

Recommended order:

1. Run `research_notebooks/<COIN>/<COIN>_Research_Regression.ipynb` to inspect factor fit and diagnostics.
2. Run `research_notebooks/<COIN>/<COIN>_Trading_Forecast_Regression.ipynb` to inspect out-of-sample forecast performance and strategy results.

## Reusable Source Code

Shared notebook logic lives under `src/`.

The main reusable modules are:

- `src/paths.py`: find the project root and data file paths.
- `src/load_data.py`: load Binance, FRED, yfinance, funding, DVOL, returns, and cumulative returns.
- `src/features.py`: build same-hour research features or lagged trading features for any target coin.
- `src/modeling.py`: fit OLS, Ridge, and ElasticNet model comparisons.
- `src/backtest.py`: calculate signal, strategy equity, and trading metrics.
- `src/plots.py`: shared cumulative-return, correlation, and regression-fit charts.

In notebooks, add `src` to `sys.path`, then import the shared helpers:

```python
import sys
from pathlib import Path

project_root = next(
    path for path in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    if (path / "refresh_data.py").exists() and (path / "data").exists()
)
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
```

Then import only the helpers needed by the notebook:

```python
from features import build_research_features, build_trading_features
from load_data import load_market_data
from modeling import fit_regression_models
from backtest import run_strategy_backtest
from plots import plot_correlation_matrix, plot_cumulative_returns, plot_regression_fit
```


## Limitations

- Yahoo Finance hourly data is market-hours data, while crypto trades 24/7.
- FRED macro series are not hourly, so notebooks forward-fill them onto hourly timestamps.
- Funding data is event-based and bucketed into the research interval.
- Backtest results do not automatically imply live trading profitability.
- This is research code, not investment advice.

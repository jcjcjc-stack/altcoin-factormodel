from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm


# Setup
project_root = Path(__file__).resolve().parent
raw_data_dir = project_root / "data" / "raw" / "cryptocompare_daily"
data_path = raw_data_dir / "cryptocompare_daily_ohlcv_combined.csv"
fred_data_path = project_root / "data" / "raw" / "fred_macro" / "fred_macro_combined.csv"
yfinance_data_path = project_root / "data" / "raw" / "yfinance_market" / "yfinance_market_ohlcv_combined.csv"
btc_funding_path = project_root / "data" / "raw" / "binance_funding" / "BTCUSDT_funding_rate_daily.csv"
btc_dvol_path = project_root / "data" / "raw" / "deribit_dvol" / "BTC_DVOL_1d.csv"
research_output_dir = project_root / "outputs" / "research"
chart_dir = research_output_dir / "charts"
table_dir = research_output_dir / "tables"

chart_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)

coin_order = ["PEPE", "BTC", "ETH", "SOL", "DOGE", "SHIB"]
target = "PEPE"
coin_colors = {
    "PEPE": "green",
    "BTC": "orange",
    "ETH": "grey",
    "SOL": "purple",
    "DOGE": "red",
    "SHIB": "brown",
}


# Load downloaded CryptoCompare OHLCV data
df = pd.read_csv(data_path, parse_dates=["timestamp"])
df["coin"] = df["symbol"].str.replace("USD", "", regex=False)

numeric_columns = ["open", "high", "low", "close", "volume_from"]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

bad_pepe_price_rows = df["coin"].eq("PEPE") & df["close"].gt(0.001)
if bad_pepe_price_rows.any():
    df.loc[bad_pepe_price_rows].to_csv(table_dir / "pepe_research_excluded_bad_cryptocompare_rows.csv", index=False)
    df.loc[bad_pepe_price_rows, numeric_columns] = np.nan

close_prices = df.pivot(index="timestamp", columns="coin", values="close").sort_index()
close_prices = close_prices[coin_order].replace([np.inf, -np.inf], np.nan)

open_prices = df.pivot(index="timestamp", columns="coin", values="open").sort_index()
high_prices = df.pivot(index="timestamp", columns="coin", values="high").sort_index()
low_prices = df.pivot(index="timestamp", columns="coin", values="low").sort_index()
volume_from = df.pivot(index="timestamp", columns="coin", values="volume_from").sort_index()

open_prices = open_prices[coin_order].replace([np.inf, -np.inf], np.nan)
high_prices = high_prices[coin_order].replace([np.inf, -np.inf], np.nan)
low_prices = low_prices[coin_order].replace([np.inf, -np.inf], np.nan)
volume_from = volume_from[coin_order].replace([np.inf, -np.inf], np.nan)


# Build returns and CryptoCompare-derived factors
raw_returns = close_prices.pct_change(fill_method=None)
raw_returns = raw_returns[coin_order].replace([np.inf, -np.inf], np.nan)
for coin in coin_order:
    first_valid_timestamp = close_prices[coin].first_valid_index()
    if first_valid_timestamp is not None:
        raw_returns.loc[first_valid_timestamp, coin] = 0.0
raw_returns.to_csv(raw_data_dir / "daily_raw_returns.csv")

log_returns = np.log(close_prices / close_prices.shift(1))
log_returns = log_returns[coin_order].replace([np.inf, -np.inf], np.nan)
for coin in coin_order:
    first_valid_timestamp = close_prices[coin].first_valid_index()
    if first_valid_timestamp is not None:
        log_returns.loc[first_valid_timestamp, coin] = 0.0
log_returns.to_csv(raw_data_dir / "daily_log_returns.csv")

fred_data = pd.read_csv(fred_data_path, parse_dates=["date"])
fred_data["value"] = pd.to_numeric(fred_data["value"], errors="coerce")
fred_features = fred_data.pivot(index="date", columns="factor", values="value").sort_index()
fred_features.index = pd.to_datetime(fred_features.index, utc=True)
fred_features = fred_features.reindex(close_prices.index).ffill().replace([np.inf, -np.inf], np.nan)

yfinance_data = pd.read_csv(yfinance_data_path, parse_dates=["date"])
yfinance_data["close"] = pd.to_numeric(yfinance_data["close"], errors="coerce")
yfinance_closes = yfinance_data.pivot(index="date", columns="factor", values="close").sort_index()
yfinance_closes.index = pd.to_datetime(yfinance_closes.index, utc=True)
yfinance_closes = yfinance_closes.reindex(close_prices.index).ffill().replace([np.inf, -np.inf], np.nan)
yfinance_log_returns = np.log(yfinance_closes / yfinance_closes.shift(1)).replace([np.inf, -np.inf], np.nan)

btc_funding = pd.read_csv(btc_funding_path, parse_dates=["date"])
btc_funding["funding_rate_sum"] = pd.to_numeric(btc_funding["funding_rate_sum"], errors="coerce")
btc_funding = btc_funding.set_index("date").sort_index()
btc_funding.index = pd.to_datetime(btc_funding.index, utc=True)
btc_funding = btc_funding.reindex(close_prices.index).ffill().replace([np.inf, -np.inf], np.nan)

btc_dvol = pd.read_csv(btc_dvol_path, parse_dates=["date"])
btc_dvol["close"] = pd.to_numeric(btc_dvol["close"], errors="coerce")
btc_dvol = btc_dvol.set_index("date").sort_index()
btc_dvol.index = pd.to_datetime(btc_dvol.index, utc=True)
btc_dvol = btc_dvol.reindex(close_prices.index).ffill().replace([np.inf, -np.inf], np.nan)

cumulative_returns = np.exp(log_returns.cumsum()) - 1
cumulative_returns.to_csv(table_dir / "pepe_research_cumulative_log_return_index.csv")

simple_cumulative_returns = (1 + raw_returns).cumprod() - 1
simple_cumulative_returns.to_csv(table_dir / "pepe_research_simple_cumulative_return_index.csv")

print("Simple cumulative return:")
print(simple_cumulative_returns.tail())
print()

fig, ax = plt.subplots(figsize=(12, 6))
simple_cumulative_returns.plot(
    ax=ax,
    linewidth=1.8,
    color=[coin_colors.get(coin) for coin in simple_cumulative_returns.columns],
)

chart_start = simple_cumulative_returns.index.min()
chart_end = simple_cumulative_returns.index.max()
event_y = simple_cumulative_returns.max().max()

event_spans = [
    {
        "label": "COVID shock",
        "start": "2020-03-11",
        "end": "2020-04-30",
        "color": "lightgrey",
    },
    {
        "label": "BTC ETF build-up",
        "start": "2023-06-15",
        "end": "2024-01-10",
        "color": "gold",
    },
    {
        "label": "Tariff stress",
        "start": "2025-04-02",
        "end": "2025-04-30",
        "color": "lightcoral",
    },
    {
        "label": "Q4 crypto deleveraging",
        "start": "2025-10-10",
        "end": "2025-12-31",
        "color": "lightgrey",
    },
]

event_lines = [
    {"label": "SVB collapse", "date": "2023-03-10", "color": "black"},
    {"label": "Israel-Hamas war", "date": "2023-10-07", "color": "darkred"},
    {"label": "BTC ETF approved", "date": "2024-01-10", "color": "black"},
    {"label": "BTC halving", "date": "2024-04-20", "color": "orange"},
    {"label": "US election", "date": "2024-11-05", "color": "blue"},
    {"label": "Trump inauguration", "date": "2025-01-20", "color": "black"},
]

for event in event_spans:
    start = pd.Timestamp(event["start"], tz="UTC")
    end = pd.Timestamp(event["end"], tz="UTC")
    if end < chart_start or start > chart_end:
        continue

    visible_start = max(start, chart_start)
    visible_end = min(end, chart_end)
    ax.axvspan(visible_start, visible_end, color=event["color"], alpha=0.14)
    ax.text(
        visible_start,
        event_y,
        event["label"],
        rotation=90,
        va="top",
        ha="left",
        fontsize=8,
        color="black",
    )

for event in event_lines:
    date = pd.Timestamp(event["date"], tz="UTC")
    if date < chart_start or date > chart_end:
        continue

    ax.axvline(date, color=event["color"], linewidth=1.0, linestyle="--", alpha=0.8)
    ax.text(
        date,
        event_y,
        event["label"],
        rotation=90,
        va="top",
        ha="right",
        fontsize=8,
        color=event["color"],
    )

plt.title("Simple Cumulative Return")
plt.xlabel("Date")
plt.ylabel("Simple cumulative return")
plt.grid(True, alpha=0.3)
plt.legend(title="Coin", ncols=4)
plt.tight_layout()
plt.savefig(chart_dir / "pepe_research_simple_cumulative_return_index.png", dpi=160)
plt.show()


# Build same-day feature matrix for accuracy testing
# - target PEPE is today's log return
# - predictors are same-day market log returns
# - PEPE and BTC technical factors are lagged so they use information known before today
features = pd.DataFrame(index=log_returns.index)
features["BTC"] = log_returns["BTC"]
features["ETH"] = log_returns["ETH"]
features["SOL"] = log_returns["SOL"]
features["DOGE"] = log_returns["DOGE"]
features["SHIB"] = log_returns["SHIB"]
features["PEPE_lag1"] = log_returns["PEPE"].shift(1)
features["PEPE_lag2"] = log_returns["PEPE"].shift(2)
features["PEPE_lag3"] = log_returns["PEPE"].shift(3)
features["PEPE_lag5"] = log_returns["PEPE"].shift(5)
features["PEPE_momentum_7d_lag1"] = log_returns["PEPE"].rolling(7).sum().shift(1)
features["PEPE_momentum_14d_lag1"] = log_returns["PEPE"].rolling(14).sum().shift(1)
features["PEPE_volatility_7d_lag1"] = log_returns["PEPE"].rolling(7).std().shift(1)
features["PEPE_volatility_14d_lag1"] = log_returns["PEPE"].rolling(14).std().shift(1)
features["PEPE_volume_from_change_lag1"] = np.log(volume_from["PEPE"] / volume_from["PEPE"].shift(1)).shift(1)
features["PEPE_range_lag1"] = ((high_prices["PEPE"] - low_prices["PEPE"]) / close_prices["PEPE"]).shift(1)
features["BTC_volatility_14d_lag1"] = log_returns["BTC"].rolling(14).std().shift(1)
features["BTC_range_lag1"] = ((high_prices["BTC"] - low_prices["BTC"]) / close_prices["BTC"]).shift(1)
features["BTC_funding_rate_sum_lag1"] = btc_funding["funding_rate_sum"].shift(1)
features["BTC_DVOL_change"] = btc_dvol["close"].diff()
features["M2_30d_change"] = np.log(fred_features["m2_money_supply"] / fred_features["m2_money_supply"].shift(30))
features["SOFR_change"] = fred_features["sofr"].diff()
features["DFF_change"] = fred_features["effective_fed_funds_rate"].diff()
features["US_10Y_change"] = fred_features["us_10y_treasury_yield"].diff()
features["VIX_change"] = np.log(fred_features["vix_close"] / fred_features["vix_close"].shift(1))
features["High_yield_spread_change"] = fred_features["high_yield_spread"].diff()
features["Dollar_index_change"] = np.log(fred_features["broad_us_dollar_index"] / fred_features["broad_us_dollar_index"].shift(1))
features["QQQ"] = yfinance_log_returns["nasdaq_100_etf"]
features["SPY"] = yfinance_log_returns["sp500_etf"]
features["GLD"] = yfinance_log_returns["gold_etf"]
features["SLV"] = yfinance_log_returns["silver_etf"]

model_data = pd.concat([log_returns[target].rename(target), features], axis=1).dropna()
model_data.to_csv(table_dir / "pepe_research_feature_matrix.csv")

print("Model data:")
print(model_data.tail())
print()
print("Dependent Variable is today's PEPE log return.")
print("Predictors use CryptoCompare daily OHLCV data.")
print("Market return factors are same-day returns.")
print("PEPE/BTC technical factors are lagged one day to avoid using today's PEPE close-derived information.")


# Correlation matrix
correlation_matrix = model_data.corr()
correlation_matrix.to_csv(table_dir / "pepe_research_feature_correlation_matrix.csv")

plt.figure(figsize=(11, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
plt.title("PEPE CryptoCompare Feature Correlation Matrix")

for row_index, row_name in enumerate(correlation_matrix.index):
    for column_index, column_name in enumerate(correlation_matrix.columns):
        value = correlation_matrix.loc[row_name, column_name]
        plt.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)

plt.tight_layout()
plt.savefig(chart_dir / "pepe_research_feature_correlation_matrix.png", dpi=160)
plt.show()


# Linear regression
feature_columns = list(features.columns)
split_idx = int(len(model_data) * 0.8)

train_data = model_data.iloc[:split_idx]
test_data = model_data.iloc[split_idx:]

X_train = sm.add_constant(train_data[feature_columns], has_constant="add")
y_train = train_data[target]
X_test = sm.add_constant(test_data[feature_columns], has_constant="add")
y_test = test_data[target]

model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_test)

train_r2 = model.rsquared
test_r2 = r2_score(y_test, y_pred)
adjusted_test_r2 = 1 - (1 - test_r2) * (len(y_test) - 1) / (len(y_test) - len(feature_columns) - 1)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
pred_corr = np.corrcoef(y_test, y_pred)[0, 1]

cv_r2s = []
for train_idx, val_idx in TimeSeriesSplit(n_splits=5).split(train_data):
    cv_X_train = sm.add_constant(train_data.iloc[train_idx][feature_columns], has_constant="add")
    cv_y_train = train_data.iloc[train_idx][target]
    cv_X_val = sm.add_constant(train_data.iloc[val_idx][feature_columns], has_constant="add")
    cv_y_val = train_data.iloc[val_idx][target]
    cv_model = sm.OLS(cv_y_train, cv_X_train).fit()
    cv_pred = cv_model.predict(cv_X_val)
    cv_r2s.append(r2_score(cv_y_val, cv_pred))

coefficients = model.params.rename("coefficient")

coefficients.to_csv(table_dir / "pepe_research_coefficients.csv")
summary_path = table_dir / "pepe_research_ols_summary.txt"
summary_path.write_text(model.summary().as_text(), encoding="utf-8")

print()
print(model.summary())


# Regression chart
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(test_data.index, y_test, label="Actual PEPE", color="green", linewidth=1.2)
axes[0].plot(test_data.index, y_pred, label="Predicted PEPE", color="black", linestyle="--")
axes[0].set_title(
    f"PEPE CryptoCompare Linear Regression: Actual vs Fitted "
    f"(Train R2={train_r2:.3f}, Test R2={test_r2:.3f})"
)
axes[0].set_ylabel("Log return")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

residuals = y_test - y_pred
axes[1].scatter(test_data.index, residuals, s=8, color="steelblue", alpha=0.6)
axes[1].axhline(0, color="red", linewidth=0.8, linestyle="--")
axes[1].set_title("Residuals: Actual minus Predicted")
axes[1].set_ylabel("Residual")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(chart_dir / "pepe_research_actual_vs_fitted.png", dpi=160)
plt.show()

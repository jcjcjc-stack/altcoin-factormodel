from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


# Setup
project_root = Path(__file__).resolve().parent
raw_data_dir = project_root / "data" / "raw" / "cryptocompare_daily"
data_path = raw_data_dir / "cryptocompare_daily_ohlcv_combined.csv"
fred_data_path = project_root / "data" / "raw" / "fred_macro" / "fred_macro_combined.csv"
yfinance_data_path = project_root / "data" / "raw" / "yfinance_market" / "yfinance_market_ohlcv_combined.csv"
btc_funding_path = project_root / "data" / "raw" / "binance_funding" / "BTCUSDT_funding_rate_daily.csv"
btc_dvol_path = project_root / "data" / "raw" / "deribit_dvol" / "BTC_DVOL_1d.csv"
forecast_output_dir = project_root / "outputs" / "trading_forecast"
chart_dir = forecast_output_dir / "charts"
table_dir = forecast_output_dir / "tables"

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
    df.loc[bad_pepe_price_rows].to_csv(table_dir / "pepe_forecast_excluded_bad_cryptocompare_rows.csv", index=False)
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
cumulative_returns.to_csv(table_dir / "pepe_forecast_cumulative_log_return_index.csv")

simple_cumulative_returns = (1 + raw_returns).cumprod() - 1
simple_cumulative_returns.to_csv(table_dir / "pepe_forecast_simple_cumulative_return_index.csv")

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
plt.savefig(chart_dir / "pepe_forecast_simple_cumulative_return_index.png", dpi=160)
plt.show()


# Build lagged feature matrix for trading-style forecast testing
# - target PEPE is today's log return
# - predictors are lagged so they use information known before today
features = pd.DataFrame(index=log_returns.index)
# Crypto features
features["BTC_lag1"] = log_returns["BTC"].shift(1)
features["ETH_lag1"] = log_returns["ETH"].shift(1)
features["SOL_lag1"] = log_returns["SOL"].shift(1)
features["DOGE_lag1"] = log_returns["DOGE"].shift(1)
features["SHIB_lag1"] = log_returns["SHIB"].shift(1)
# PEPE features
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
# BTC features
features["BTC_volatility_14d_lag1"] = log_returns["BTC"].rolling(14).std().shift(1)
features["BTC_range_lag1"] = ((high_prices["BTC"] - low_prices["BTC"]) / close_prices["BTC"]).shift(1)
features["BTC_funding_rate_sum_lag1"] = btc_funding["funding_rate_sum"].shift(1)
features["BTC_DVOL_change_lag1"] = btc_dvol["close"].diff().shift(1)
# Macro features
features["M2_30d_change_lag1"] = np.log(fred_features["m2_money_supply"] / fred_features["m2_money_supply"].shift(30)).shift(1)
features["SOFR_change_lag1"] = fred_features["sofr"].diff().shift(1)
features["DFF_change_lag1"] = fred_features["effective_fed_funds_rate"].diff().shift(1)
features["US_10Y_change_lag1"] = fred_features["us_10y_treasury_yield"].diff().shift(1)
features["VIX_change_lag1"] = np.log(fred_features["vix_close"] / fred_features["vix_close"].shift(1)).shift(1)
features["High_yield_spread_change_lag1"] = fred_features["high_yield_spread"].diff().shift(1)
features["Dollar_index_change_lag1"] = np.log(fred_features["broad_us_dollar_index"] / fred_features["broad_us_dollar_index"].shift(1)).shift(1)
# ETF features
features["QQQ_lag1"] = yfinance_log_returns["nasdaq_100_etf"].shift(1)
features["SPY_lag1"] = yfinance_log_returns["sp500_etf"].shift(1)
features["GLD_lag1"] = yfinance_log_returns["gold_etf"].shift(1)
features["SLV_lag1"] = yfinance_log_returns["silver_etf"].shift(1)
# Technical Indicators
trading_indicators = pd.DataFrame(index=log_returns.index)
trading_indicators["PEPE_SMA12"] = log_returns["PEPE"].rolling(12).mean().shift(1)
trading_indicators["PEPE_SMA26"] = log_returns["PEPE"].rolling(26).mean().shift(1)
pepe_close_change = close_prices["PEPE"].diff()
pepe_gain = pepe_close_change.clip(lower=0)
pepe_loss = -pepe_close_change.clip(upper=0)
pepe_avg_gain_14d = pepe_gain.rolling(14).mean()
pepe_avg_loss_14d = pepe_loss.rolling(14).mean()
trading_indicators["PEPE_RSI_14d_lag1"] = (100 - (100 / (1 + pepe_avg_gain_14d / pepe_avg_loss_14d))).shift(1)

pepe_bb_middle_20d = close_prices["PEPE"].rolling(20).mean()
pepe_bb_std_20d = close_prices["PEPE"].rolling(20).std()
pepe_bb_upper_20d = pepe_bb_middle_20d + 2 * pepe_bb_std_20d
pepe_bb_lower_20d = pepe_bb_middle_20d - 2 * pepe_bb_std_20d
trading_indicators["PEPE_BB_percent_b_20d_lag1"] = (
    (close_prices["PEPE"] - pepe_bb_lower_20d) / (pepe_bb_upper_20d - pepe_bb_lower_20d)
).shift(1)
features = pd.concat([features, trading_indicators], axis=1)

model_data = pd.concat([log_returns[target].rename(target), features], axis=1).dropna()
trading_indicator_table = trading_indicators.reindex(model_data.index)
trading_indicator_table.to_csv(table_dir / "pepe_forecast_trading_indicators.csv")
model_data.to_csv(table_dir / "pepe_forecast_feature_matrix.csv")

print("Model data:")
print(model_data.tail())
print()
print("Trading indicators:")
print(trading_indicator_table.tail())
print()
print("Dependent Variable is today's PEPE log return.")
print("Predictors use CryptoCompare daily OHLCV data.")
print("All predictors are lagged so they use information known before today's PEPE return.")


# Correlation matrix
correlation_matrix = model_data.corr()
correlation_matrix.to_csv(table_dir / "pepe_forecast_feature_correlation_matrix.csv")

plt.figure(figsize=(11, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
plt.title("PEPE Forecast Feature Correlation Matrix")

for row_index, row_name in enumerate(correlation_matrix.index):
    for column_index, column_name in enumerate(correlation_matrix.columns):
        value = correlation_matrix.loc[row_name, column_name]
        plt.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)

plt.tight_layout()
plt.savefig(chart_dir / "pepe_forecast_feature_correlation_matrix.png", dpi=160)
plt.show()


# Linear, Ridge, and ElasticNet regression
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
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(train_data):
    cv_X_train = sm.add_constant(train_data.iloc[train_idx][feature_columns], has_constant="add")
    cv_y_train = train_data.iloc[train_idx][target]
    cv_X_val = sm.add_constant(train_data.iloc[val_idx][feature_columns], has_constant="add")
    cv_y_val = train_data.iloc[val_idx][target]
    cv_model = sm.OLS(cv_y_train, cv_X_train).fit()
    cv_pred = cv_model.predict(cv_X_val)
    cv_r2s.append(r2_score(cv_y_val, cv_pred))

coefficients = model.params.rename("coefficient")

coefficients.to_csv(table_dir / "pepe_forecast_coefficients.csv")
coefficients.to_csv(table_dir / "pepe_forecast_ols_coefficients.csv")
summary_path = table_dir / "pepe_forecast_ols_summary.txt"
summary_path.write_text(model.summary().as_text(), encoding="utf-8")

print()
print(model.summary())

ridge_X_train = train_data[feature_columns]
ridge_X_test = test_data[feature_columns]
ridge_alphas = [0.01, 0.1, 1, 10, 100, 300, 1000, 3000, 10000]

ridge_cv_results = []

for alpha in ridge_alphas:
    alpha_cv_r2s = []
    for train_idx, val_idx in tscv.split(train_data):
        cv_X_train = train_data.iloc[train_idx][feature_columns]
        cv_y_train = train_data.iloc[train_idx][target]
        cv_X_val = train_data.iloc[val_idx][feature_columns]
        cv_y_val = train_data.iloc[val_idx][target]
        cv_ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        cv_ridge_model.fit(cv_X_train, cv_y_train)
        alpha_cv_r2s.append(r2_score(cv_y_val, cv_ridge_model.predict(cv_X_val)))
    ridge_cv_results.append({"alpha": alpha, "cv_r2_mean": np.mean(alpha_cv_r2s)})

ridge_cv_results = pd.DataFrame(ridge_cv_results)
best_ridge_alpha = ridge_cv_results.loc[ridge_cv_results["cv_r2_mean"].idxmax(), "alpha"]
ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=best_ridge_alpha))
ridge_model.fit(ridge_X_train, y_train)
ridge_y_pred = ridge_model.predict(ridge_X_test)

ridge_train_r2 = ridge_model.score(ridge_X_train, y_train)
ridge_test_r2 = r2_score(y_test, ridge_y_pred)
ridge_adjusted_test_r2 = 1 - (1 - ridge_test_r2) * (len(y_test) - 1) / (len(y_test) - len(feature_columns) - 1)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_y_pred))
ridge_pred_corr = np.corrcoef(y_test, ridge_y_pred)[0, 1]

ridge_coefficients = pd.Series(
    ridge_model.named_steps["ridge"].coef_,
    index=feature_columns,
    name="coefficient",
)
ridge_coefficients.to_csv(table_dir / "pepe_forecast_ridge_coefficients.csv")
ridge_cv_results.to_csv(table_dir / "pepe_forecast_ridge_alpha_cv.csv", index=False)

elastic_net_alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
elastic_net_l1_ratios = [0.1, 0.5, 0.9]
elastic_net_cv_results = []

for alpha in elastic_net_alphas:
    for l1_ratio in elastic_net_l1_ratios:
        alpha_cv_r2s = []
        for train_idx, val_idx in tscv.split(train_data):
            cv_X_train = train_data.iloc[train_idx][feature_columns]
            cv_y_train = train_data.iloc[train_idx][target]
            cv_X_val = train_data.iloc[val_idx][feature_columns]
            cv_y_val = train_data.iloc[val_idx][target]
            cv_elastic_net_model = make_pipeline(
                StandardScaler(),
                ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000),
            )
            cv_elastic_net_model.fit(cv_X_train, cv_y_train)
            alpha_cv_r2s.append(r2_score(cv_y_val, cv_elastic_net_model.predict(cv_X_val)))
        elastic_net_cv_results.append(
            {
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "cv_r2_mean": np.mean(alpha_cv_r2s),
            }
        )

elastic_net_cv_results = pd.DataFrame(elastic_net_cv_results)
best_elastic_net_params = elastic_net_cv_results.loc[elastic_net_cv_results["cv_r2_mean"].idxmax()]
elastic_net_model = make_pipeline(
    StandardScaler(),
    ElasticNet(
        alpha=best_elastic_net_params["alpha"],
        l1_ratio=best_elastic_net_params["l1_ratio"],
        max_iter=100000,
    ),
)
elastic_net_model.fit(ridge_X_train, y_train)
elastic_net_y_pred = elastic_net_model.predict(ridge_X_test)

elastic_net_train_r2 = elastic_net_model.score(ridge_X_train, y_train)
elastic_net_test_r2 = r2_score(y_test, elastic_net_y_pred)
elastic_net_adjusted_test_r2 = 1 - (1 - elastic_net_test_r2) * (len(y_test) - 1) / (len(y_test) - len(feature_columns) - 1)
elastic_net_rmse = np.sqrt(mean_squared_error(y_test, elastic_net_y_pred))
elastic_net_pred_corr = np.corrcoef(y_test, elastic_net_y_pred)[0, 1]
elastic_net_nonzero_coefficients = np.count_nonzero(elastic_net_model.named_steps["elasticnet"].coef_)

elastic_net_coefficients = pd.Series(
    elastic_net_model.named_steps["elasticnet"].coef_,
    index=feature_columns,
    name="coefficient",
)
elastic_net_coefficients.to_csv(table_dir / "pepe_forecast_elastic_net_coefficients.csv")
elastic_net_cv_results.to_csv(table_dir / "pepe_forecast_elastic_net_cv.csv", index=False)

annual_periods = 365
strategy_metric_rows = []
strategy_series = {}
prediction_sets = {
    "OLS": pd.Series(y_pred, index=y_test.index),
    "Ridge": pd.Series(ridge_y_pred, index=y_test.index),
    "ElasticNet": pd.Series(elastic_net_y_pred, index=y_test.index),
}

for model_name, predicted_returns in prediction_sets.items():
    signal = np.sign(predicted_returns).replace(0, np.nan).ffill().fillna(0)
    trade_events = signal.ne(signal.shift(1).fillna(0))
    strategy_log_returns = signal * y_test
    strategy_equity = np.exp(strategy_log_returns.cumsum())
    strategy_drawdown = strategy_equity / strategy_equity.cummax() - 1
    downside_returns = strategy_log_returns[strategy_log_returns < 0]
    daily_std = strategy_log_returns.std()
    downside_std = downside_returns.std()
    strategy_series[model_name] = pd.DataFrame(
        {
            "actual_log_return": y_test,
            "predicted_log_return": predicted_returns,
            "signal": signal,
            "trade_event": trade_events,
            "strategy_log_return": strategy_log_returns,
            "strategy_equity": strategy_equity,
            "strategy_cumulative_pnl": strategy_equity - 1,
            "strategy_drawdown": strategy_drawdown,
        }
    )

    strategy_metric_rows.append(
        {
            "model": model_name,
            "number_of_trades": int(trade_events.sum()),
            "cagr": np.exp(strategy_log_returns.mean() * annual_periods) - 1,
            "max_drawdown": strategy_drawdown.min(),
            "sharpe_ratio": strategy_log_returns.mean() / daily_std * np.sqrt(annual_periods) if daily_std > 0 else np.nan,
            "sortino_ratio": (
                strategy_log_returns.mean() / downside_std * np.sqrt(annual_periods) if downside_std > 0 else np.nan
            ),
            "directional_accuracy": (np.sign(predicted_returns) == np.sign(y_test)).mean(),
        }
    )

strategy_metrics = pd.DataFrame(strategy_metric_rows).set_index("model")
strategy_metrics.to_csv(table_dir / "pepe_forecast_strategy_metrics.csv")
pd.concat(strategy_series, names=["model", "timestamp"]).to_csv(table_dir / "pepe_forecast_strategy_timeseries.csv")

model_comparison = pd.DataFrame(
    [
        {
            "model": "OLS",
            "alpha": np.nan,
            "l1_ratio": np.nan,
            "nonzero_coefficients": len(feature_columns),
            "number_of_trades": strategy_metrics.loc["OLS", "number_of_trades"],
            "cagr": strategy_metrics.loc["OLS", "cagr"],
            "max_drawdown": strategy_metrics.loc["OLS", "max_drawdown"],
            "sharpe_ratio": strategy_metrics.loc["OLS", "sharpe_ratio"],
            "sortino_ratio": strategy_metrics.loc["OLS", "sortino_ratio"],
            "directional_accuracy": strategy_metrics.loc["OLS", "directional_accuracy"],
            "train_r2": train_r2,
            "test_r2": test_r2,
            "adjusted_test_r2": adjusted_test_r2,
            "rmse": rmse,
            "prediction_correlation": pred_corr,
            "cv_r2_mean": np.mean(cv_r2s),
        },
        {
            "model": "Ridge",
            "alpha": best_ridge_alpha,
            "l1_ratio": np.nan,
            "nonzero_coefficients": np.count_nonzero(ridge_model.named_steps["ridge"].coef_),
            "number_of_trades": strategy_metrics.loc["Ridge", "number_of_trades"],
            "cagr": strategy_metrics.loc["Ridge", "cagr"],
            "max_drawdown": strategy_metrics.loc["Ridge", "max_drawdown"],
            "sharpe_ratio": strategy_metrics.loc["Ridge", "sharpe_ratio"],
            "sortino_ratio": strategy_metrics.loc["Ridge", "sortino_ratio"],
            "directional_accuracy": strategy_metrics.loc["Ridge", "directional_accuracy"],
            "train_r2": ridge_train_r2,
            "test_r2": ridge_test_r2,
            "adjusted_test_r2": ridge_adjusted_test_r2,
            "rmse": ridge_rmse,
            "prediction_correlation": ridge_pred_corr,
            "cv_r2_mean": ridge_cv_results["cv_r2_mean"].max(),
        },
        {
            "model": "ElasticNet",
            "alpha": best_elastic_net_params["alpha"],
            "l1_ratio": best_elastic_net_params["l1_ratio"],
            "nonzero_coefficients": elastic_net_nonzero_coefficients,
            "number_of_trades": strategy_metrics.loc["ElasticNet", "number_of_trades"],
            "cagr": strategy_metrics.loc["ElasticNet", "cagr"],
            "max_drawdown": strategy_metrics.loc["ElasticNet", "max_drawdown"],
            "sharpe_ratio": strategy_metrics.loc["ElasticNet", "sharpe_ratio"],
            "sortino_ratio": strategy_metrics.loc["ElasticNet", "sortino_ratio"],
            "directional_accuracy": strategy_metrics.loc["ElasticNet", "directional_accuracy"],
            "train_r2": elastic_net_train_r2,
            "test_r2": elastic_net_test_r2,
            "adjusted_test_r2": elastic_net_adjusted_test_r2,
            "rmse": elastic_net_rmse,
            "prediction_correlation": elastic_net_pred_corr,
            "cv_r2_mean": elastic_net_cv_results["cv_r2_mean"].max(),
        },
    ]
)
model_comparison.to_csv(table_dir / "pepe_forecast_model_comparison.csv", index=False)

model_accuracy_comparison = model_comparison[
    [
        "model",
        "alpha",
        "l1_ratio",
        "nonzero_coefficients",
        "train_r2",
        "test_r2",
        "adjusted_test_r2",
        "rmse",
        "prediction_correlation",
        "cv_r2_mean",
    ]
]
model_trade_comparison = model_comparison[
    [
        "model",
        "number_of_trades",
        "cagr",
        "max_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "directional_accuracy",
    ]
]
model_accuracy_comparison.to_csv(table_dir / "pepe_forecast_model_accuracy_comparison.csv", index=False)
model_trade_comparison.to_csv(table_dir / "pepe_forecast_model_trade_comparison.csv", index=False)

print()
print("Model accuracy comparison")
print(model_accuracy_comparison.to_string(index=False))
print()
print("Model trade comparison")
print(model_trade_comparison.to_string(index=False))


pepe_test_cumulative_returns = simple_cumulative_returns["PEPE"].reindex(y_test.index)

for model_name, model_strategy in strategy_series.items():
    model_file_name = model_name.lower().replace(" ", "_")
    trade_log = model_strategy.loc[model_strategy["trade_event"]].copy()
    trade_log["new_position"] = trade_log["signal"].map({1.0: "Long", -1.0: "Short", 0.0: "Flat"})
    trade_log.to_csv(table_dir / f"pepe_forecast_{model_file_name}_trade_log.csv")

    long_trades = trade_log[trade_log["signal"].eq(1)]
    short_trades = trade_log[trade_log["signal"].eq(-1)]
    flat_trades = trade_log[trade_log["signal"].eq(0)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(
        pepe_test_cumulative_returns.index,
        pepe_test_cumulative_returns,
        label="PEPE cumulative return",
        color="green",
        linewidth=1.6,
    )
    axes[0].scatter(
        long_trades.index,
        pepe_test_cumulative_returns.reindex(long_trades.index),
        marker="^",
        color="blue",
        s=45,
        label="Long trade",
        zorder=3,
    )
    axes[0].scatter(
        short_trades.index,
        pepe_test_cumulative_returns.reindex(short_trades.index),
        marker="v",
        color="red",
        s=45,
        label="Short trade",
        zorder=3,
    )
    if not flat_trades.empty:
        axes[0].scatter(
            flat_trades.index,
            pepe_test_cumulative_returns.reindex(flat_trades.index),
            marker="o",
            color="black",
            s=28,
            label="Flat trade",
            zorder=3,
        )
    axes[0].set_title(f"PEPE Test Cumulative Return with {model_name} Trades")
    axes[0].set_ylabel("PEPE cumulative return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        model_strategy.index,
        model_strategy["strategy_cumulative_pnl"],
        label=f"{model_name} cumulative PnL",
        color="black",
        linewidth=1.6,
    )
    axes[1].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[1].set_title(f"{model_name} Strategy PnL Over Time")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Cumulative PnL")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(chart_dir / f"pepe_forecast_{model_file_name}_trades_and_pnl.png", dpi=160)
    plt.show()


# Regression chart
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(test_data.index, y_test, label="Actual PEPE", color="green", linewidth=1.2)
axes[0].plot(test_data.index, y_pred, label="OLS predicted PEPE", color="black", linestyle="--")
axes[0].plot(test_data.index, ridge_y_pred, label="Ridge predicted PEPE", color="blue", linestyle=":")
axes[0].plot(test_data.index, elastic_net_y_pred, label="ElasticNet predicted PEPE", color="purple", linestyle="-.")
axes[0].set_title(
    f"PEPE Trading Forecast Regression: Actual vs Fitted "
    f"(OLS Test R2={test_r2:.3f}, Ridge Test R2={ridge_test_r2:.3f}, "
    f"ElasticNet Test R2={elastic_net_test_r2:.3f})"
)
axes[0].set_ylabel("Log return")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

residuals = y_test - y_pred
ridge_residuals = y_test - ridge_y_pred
elastic_net_residuals = y_test - elastic_net_y_pred
axes[1].scatter(test_data.index, residuals, s=8, color="steelblue", alpha=0.6, label="OLS residual")
axes[1].scatter(test_data.index, ridge_residuals, s=8, color="orange", alpha=0.5, label="Ridge residual")
axes[1].scatter(test_data.index, elastic_net_residuals, s=8, color="purple", alpha=0.5, label="ElasticNet residual")
axes[1].axhline(0, color="red", linewidth=0.8, linestyle="--")
axes[1].set_title("Residuals: Actual minus Predicted")
axes[1].set_ylabel("Residual")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(chart_dir / "pepe_forecast_actual_vs_fitted.png", dpi=160)
plt.show()

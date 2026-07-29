import numpy as np
import pandas as pd


def validate_target_and_peers(target, peer_coins):
    if target in peer_coins:
        raise ValueError(f"target {target} must not be included in peer_coins.")
    if len(peer_coins) != len(set(peer_coins)):
        raise ValueError("peer_coins contains duplicate coins.")


def build_research_features(target, peer_coins, market_data):
    validate_target_and_peers(target, peer_coins)

    log_returns = market_data["log_returns"]
    close_prices = market_data["close_prices"]
    high_prices = market_data["high_prices"]
    low_prices = market_data["low_prices"]
    volume_from = market_data["volume_from"]
    fred_features = market_data["fred_features"]
    yfinance_log_returns = market_data["yfinance_log_returns"]
    btc_funding = market_data["btc_funding"]
    btc_dvol = market_data["btc_dvol"]

    features = pd.DataFrame(index=log_returns.index)
    for coin in peer_coins:
        features[coin] = log_returns[coin]

    features[f"{target}_momentum_7h"] = log_returns[target].rolling(7).sum()
    features[f"{target}_momentum_14h"] = log_returns[target].rolling(14).sum()
    features[f"{target}_volatility_7h"] = log_returns[target].rolling(7).std()
    features[f"{target}_volatility_14h"] = log_returns[target].rolling(14).std()
    features[f"{target}_volume_change"] = np.log(volume_from[target] / volume_from[target].shift(1))
    features[f"{target}_range"] = (high_prices[target] - low_prices[target]) / close_prices[target]

    features["BTC_volatility_14h"] = log_returns["BTC"].rolling(14).std()
    features["BTC_range"] = (high_prices["BTC"] - low_prices["BTC"]) / close_prices["BTC"]
    features["BTC_funding_rate_sum"] = btc_funding["funding_rate_sum"]

    if btc_dvol["close"].notna().sum() >= 500:
        features["BTC_DVOL_change"] = btc_dvol["close"].diff()
    else:
        print("Skipping BTC_DVOL_change: insufficient hourly overlap.")

    features["M2_30d_change"] = np.log(fred_features["m2_money_supply"] / fred_features["m2_money_supply"].shift(24 * 30))
    features["SOFR_change"] = fred_features["sofr"].diff()
    features["DFF_change"] = fred_features["effective_fed_funds_rate"].diff()

    for factor, name in [
        ("nasdaq_100_etf", "QQQ"),
        ("sp500_etf", "SPY"),
        ("treasury_7_10y_etf", "IEF"),
        ("treasury_20y_etf", "TLT"),
        ("us_dollar_index_proxy", "UUP"),
        ("high_yield_credit_proxy", "HYG"),
        ("vix_futures_proxy", "VXX"),
    ]:
        features[name] = yfinance_log_returns[factor]

    features[f"{target}_SMA12h"] = log_returns[target].rolling(12).mean()
    features[f"{target}_SMA26h"] = log_returns[target].rolling(26).mean()

    close_change = close_prices[target].diff()
    gain = close_change.clip(lower=0)
    loss = -close_change.clip(upper=0)
    avg_gain_14h = gain.rolling(14).mean()
    avg_loss_14h = loss.rolling(14).mean()
    features[f"{target}_RSI_14h"] = 100 - (100 / (1 + avg_gain_14h / avg_loss_14h))

    bb_middle_20h = close_prices[target].rolling(20).mean()
    bb_std_20h = close_prices[target].rolling(20).std()
    bb_upper_20h = bb_middle_20h + 2 * bb_std_20h
    bb_lower_20h = bb_middle_20h - 2 * bb_std_20h
    features[f"{target}_BB_percent_b_20h"] = (close_prices[target] - bb_lower_20h) / (bb_upper_20h - bb_lower_20h)

    model_data = pd.concat([log_returns[target].rename(target), features], axis=1).dropna()
    return features, model_data


def build_trading_features(target, peer_coins, market_data):
    validate_target_and_peers(target, peer_coins)

    log_returns = market_data["log_returns"]
    close_prices = market_data["close_prices"]
    high_prices = market_data["high_prices"]
    low_prices = market_data["low_prices"]
    volume_from = market_data["volume_from"]
    fred_features = market_data["fred_features"]
    yfinance_log_returns = market_data["yfinance_log_returns"]
    btc_funding = market_data["btc_funding"]
    btc_dvol = market_data["btc_dvol"]

    features = pd.DataFrame(index=log_returns.index)
    for coin in peer_coins:
        features[f"{coin}_lag1"] = log_returns[coin].shift(1)

    features[f"{target}_lag1"] = log_returns[target].shift(1)
    features[f"{target}_lag2"] = log_returns[target].shift(2)
    features[f"{target}_lag3"] = log_returns[target].shift(3)
    features[f"{target}_lag5"] = log_returns[target].shift(5)
    features[f"{target}_momentum_7h_lag1"] = log_returns[target].rolling(7).sum().shift(1)
    features[f"{target}_momentum_14h_lag1"] = log_returns[target].rolling(14).sum().shift(1)
    features[f"{target}_volatility_7h_lag1"] = log_returns[target].rolling(7).std().shift(1)
    features[f"{target}_volatility_14h_lag1"] = log_returns[target].rolling(14).std().shift(1)
    features[f"{target}_volume_change_lag1"] = np.log(volume_from[target] / volume_from[target].shift(1)).shift(1)
    features[f"{target}_range_lag1"] = ((high_prices[target] - low_prices[target]) / close_prices[target]).shift(1)

    features["BTC_volatility_14h_lag1"] = log_returns["BTC"].rolling(14).std().shift(1)
    features["BTC_range_lag1"] = ((high_prices["BTC"] - low_prices["BTC"]) / close_prices["BTC"]).shift(1)
    features["BTC_funding_rate_sum_lag1"] = btc_funding["funding_rate_sum"].shift(1)

    if btc_dvol["close"].notna().sum() >= 200:
        features["BTC_DVOL_change_lag1"] = btc_dvol["close"].diff().shift(1)
    else:
        print("Skipping BTC_DVOL_change_lag1: insufficient hourly overlap.")

    features["M2_30d_change_lag1"] = np.log(
        fred_features["m2_money_supply"] / fred_features["m2_money_supply"].shift(24 * 30)
    ).shift(1)
    features["SOFR_change_lag1"] = fred_features["sofr"].diff().shift(1)
    features["DFF_change_lag1"] = fred_features["effective_fed_funds_rate"].diff().shift(1)

    for factor, name in [
        ("nasdaq_100_etf", "QQQ_lag1"),
        ("sp500_etf", "SPY_lag1"),
        ("treasury_7_10y_etf", "IEF_lag1"),
        ("treasury_20y_etf", "TLT_lag1"),
        ("us_dollar_index_proxy", "UUP_lag1"),
        ("high_yield_credit_proxy", "HYG_lag1"),
        ("vix_futures_proxy", "VXX_lag1"),
    ]:
        features[name] = yfinance_log_returns[factor].shift(1)

    trading_indicators = pd.DataFrame(index=log_returns.index)
    trading_indicators[f"{target}_SMA12h_lag1"] = log_returns[target].rolling(12).mean().shift(1)
    trading_indicators[f"{target}_SMA26h_lag1"] = log_returns[target].rolling(26).mean().shift(1)

    close_change = close_prices[target].diff()
    gain = close_change.clip(lower=0)
    loss = -close_change.clip(upper=0)
    avg_gain_14h = gain.rolling(14).mean()
    avg_loss_14h = loss.rolling(14).mean()
    trading_indicators[f"{target}_RSI_14h_lag1"] = (100 - (100 / (1 + avg_gain_14h / avg_loss_14h))).shift(1)

    bb_middle_20h = close_prices[target].rolling(20).mean()
    bb_std_20h = close_prices[target].rolling(20).std()
    bb_upper_20h = bb_middle_20h + 2 * bb_std_20h
    bb_lower_20h = bb_middle_20h - 2 * bb_std_20h
    trading_indicators[f"{target}_BB_percent_b_20h_lag1"] = (
        (close_prices[target] - bb_lower_20h) / (bb_upper_20h - bb_lower_20h)
    ).shift(1)

    features = pd.concat([features, trading_indicators], axis=1)
    model_data = pd.concat([log_returns[target].rename(target), features], axis=1).dropna()
    return features, model_data, trading_indicators.reindex(model_data.index)

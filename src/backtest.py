import numpy as np
import pandas as pd


def run_strategy_backtest(
    y_test,
    prediction_sets,
    annual_periods=24 * 365,
    fee_bps=10.0,
    spread_bps=0.0,
    slippage_bps=5.0,
):
    strategy_metric_rows = []
    strategy_series = {}

    for model_name, predicted_returns in prediction_sets.items():
        signal = np.sign(predicted_returns).replace(0, np.nan).ffill().fillna(0)
        trade_events = signal.ne(signal.shift(1).fillna(0))
        turnover = (signal - signal.shift(1).fillna(0)).abs()
        fee_cost = turnover * (fee_bps / 10000)
        spread_cost = turnover * (spread_bps / 10000)
        slippage_cost = turnover * (slippage_bps / 10000)
        total_cost = fee_cost + spread_cost + slippage_cost
        gross_strategy_log_return = signal * y_test
        strategy_log_returns = gross_strategy_log_return - total_cost
        strategy_equity = np.exp(strategy_log_returns.cumsum())
        strategy_drawdown = strategy_equity / strategy_equity.cummax() - 1
        downside_returns = strategy_log_returns[strategy_log_returns < 0]
        return_std = strategy_log_returns.std()
        return_variance = strategy_log_returns.var()
        downside_std = downside_returns.std()

        strategy_series[model_name] = pd.DataFrame(
            {
                "actual_log_return": y_test,
                "predicted_log_return": predicted_returns,
                "signal": signal,
                "trade_event": trade_events,
                "gross_strategy_log_return": gross_strategy_log_return,
                "turnover": turnover,
                "fee_cost": fee_cost,
                "spread_cost": spread_cost,
                "slippage_cost": slippage_cost,
                "total_cost": total_cost,
                "net_strategy_log_return": strategy_log_returns,
                "strategy_log_return": strategy_log_returns,
                "strategy_equity": strategy_equity,
                "strategy_cumulative_pnl": strategy_equity - 1,
                "strategy_drawdown": strategy_drawdown,
            }
        )

        strategy_metric_rows.append(
            {
                "model": model_name,
                "trade_number": int(trade_events.sum()),
                "total_turnover": turnover.sum(),
                "total_cost": total_cost.sum(),
                "cumulative_return": strategy_equity.iloc[-1] - 1 if len(strategy_equity) else np.nan,
                "cagr": np.exp(strategy_log_returns.mean() * annual_periods) - 1,
                "sharpe_ratio": (
                    strategy_log_returns.mean() / return_std * np.sqrt(annual_periods) if return_std > 0 else np.nan
                ),
                "max_drawdown": strategy_drawdown.min(),
                "variance": return_variance,
                "sortino_ratio": (
                    strategy_log_returns.mean() / downside_std * np.sqrt(annual_periods)
                    if downside_std > 0
                    else np.nan
                ),
                "directional_accuracy": (np.sign(predicted_returns) == np.sign(y_test)).mean(),
            }
        )

    strategy_metrics = pd.DataFrame(strategy_metric_rows).set_index("model")
    model_trade_comparison = strategy_metrics.reset_index()[
        [
            "model",
            "trade_number",
            "total_turnover",
            "total_cost",
            "cumulative_return",
            "cagr",
            "sharpe_ratio",
            "max_drawdown",
            "variance",
            "sortino_ratio",
            "directional_accuracy",
        ]
    ]
    return strategy_metrics, model_trade_comparison, strategy_series


def build_cost_comparison(strategy_series):
    cost_rows = []

    for model_name, model_strategy in strategy_series.items():
        total_trades = model_strategy["trade_event"].sum()
        total_cost = model_strategy["total_cost"].sum()
        cost_rows.append(
            {
                "model": model_name,
                "trade_number": int(total_trades),
                "total_turnover": model_strategy["turnover"].sum(),
                "fee_cost": model_strategy["fee_cost"].sum(),
                "spread_cost": model_strategy["spread_cost"].sum(),
                "slippage_cost": model_strategy["slippage_cost"].sum(),
                "total_cost": total_cost,
                "avg_cost_per_trade": total_cost / total_trades if total_trades > 0 else 0,
            }
        )

    return pd.DataFrame(cost_rows).set_index("model")

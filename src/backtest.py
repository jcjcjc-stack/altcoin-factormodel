import numpy as np
import pandas as pd


def run_strategy_backtest(y_test, prediction_sets, annual_periods=24 * 365):
    strategy_metric_rows = []
    strategy_series = {}

    for model_name, predicted_returns in prediction_sets.items():
        signal = np.sign(predicted_returns).replace(0, np.nan).ffill().fillna(0)
        trade_events = signal.ne(signal.shift(1).fillna(0))
        strategy_log_returns = signal * y_test
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


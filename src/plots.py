import matplotlib.pyplot as plt
import pandas as pd


def add_event_markers(ax, index, event_y):
    chart_start = index.min()
    chart_end = index.max()
    event_spans = [
        {"label": "Tariff stress", "start": "2025-04-02", "end": "2025-04-30", "color": "lightcoral"},
        {"label": "Q4 crypto deleveraging", "start": "2025-10-10", "end": "2025-12-31", "color": "lightgrey"},
    ]
    event_lines = [
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
        ax.text(visible_start, event_y, event["label"], rotation=90, va="top", ha="left", fontsize=8, color="black")

    for event in event_lines:
        date = pd.Timestamp(event["date"], tz="UTC")
        if date < chart_start or date > chart_end:
            continue
        ax.axvline(date, color=event["color"], linewidth=1.0, linestyle="--", alpha=0.8)
        ax.text(date, event_y, event["label"], rotation=90, va="top", ha="right", fontsize=8, color=event["color"])


def plot_cumulative_returns(cumulative_returns, coin_colors, title, ylabel="Cumulative return", ncols=4):
    fig, ax = plt.subplots(figsize=(12, 6))
    cumulative_returns.plot(
        ax=ax,
        linewidth=1.8,
        color=[coin_colors.get(coin) for coin in cumulative_returns.columns],
    )
    add_event_markers(ax, cumulative_returns.index, cumulative_returns.max().max())
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Coin", ncols=ncols)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(model_data, title):
    correlation_matrix = model_data.corr()
    plt.figure(figsize=(11, 8))
    plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    plt.title(title)

    for row_index, _ in enumerate(correlation_matrix.index):
        for column_index, _ in enumerate(correlation_matrix.columns):
            value = correlation_matrix.iloc[row_index, column_index]
            plt.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_regression_fit(test_data, y_test, predictions, target, target_color, title):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(test_data.index, y_test, label=f"Actual {target}", color=target_color, linewidth=1.8)
    axes[0].plot(test_data.index, predictions["OLS"], label=f"OLS predicted {target}", color="black", linestyle="--", linewidth=1.2)
    axes[0].plot(test_data.index, predictions["Ridge"], label=f"Ridge predicted {target}", color="darkorange", linestyle=":", linewidth=1.4)
    axes[0].plot(test_data.index, predictions["ElasticNet"], label=f"ElasticNet predicted {target}", color="tab:cyan", linestyle="-.", linewidth=1.4)
    axes[0].set_title(title)
    axes[0].set_ylabel("Log return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for model_name, predicted in predictions.items():
        residuals = y_test - predicted
        axes[1].scatter(test_data.index, residuals, s=8, alpha=0.5, label=f"{model_name} residual")

    axes[1].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[1].set_title("Residuals: Actual minus Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


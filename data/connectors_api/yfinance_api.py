import yfinance as yf


def fetch_daily_market_rows(ticker, factor_name, start_date):
    data = yf.download(
        ticker,
        start=start_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        return []

    data = data.reset_index()
    data.columns = [
        "_".join(str(part) for part in column if part).lower().replace(" ", "_")
        if isinstance(column, tuple)
        else str(column).lower().replace(" ", "_")
        for column in data.columns
    ]
    date_column = data.columns[0]
    price_columns = {
        name: next(
            (column for column in data.columns if column == name or column.startswith(f"{name}_")),
            "",
        )
        for name in ["open", "high", "low", "close", "adj_close", "volume"]
    }

    rows = []
    for row in data.to_dict("records"):
        rows.append(
            {
                "date": row[date_column].date().isoformat(),
                "ticker": ticker,
                "factor": factor_name,
                "open": row.get(price_columns["open"], ""),
                "high": row.get(price_columns["high"], ""),
                "low": row.get(price_columns["low"], ""),
                "close": row.get(price_columns["close"], ""),
                "adj_close": row.get(price_columns["adj_close"], ""),
                "volume": row.get(price_columns["volume"], ""),
            }
        )

    return rows

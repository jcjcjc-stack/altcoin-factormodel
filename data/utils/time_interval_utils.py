from datetime import datetime


SUPPORTED_TIME_INTERVALS = {"1H", "1D", "1M", "1Y"}


def validate_time_interval(interval):
    if interval not in SUPPORTED_TIME_INTERVALS:
        raise ValueError('RESEARCH_TIME_INTERVAL must be one of "1H", "1D", "1M", or "1Y"')


def bucket_value(value, interval, as_timestamp=False):
    validate_time_interval(interval)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    if interval == "1H":
        bucket = parsed.replace(minute=0, second=0, microsecond=0).isoformat()
    elif interval == "1D":
        bucket = parsed.date().isoformat()
    elif interval == "1M":
        bucket = f"{parsed.year:04d}-{parsed.month:02d}-01"
    else:
        bucket = f"{parsed.year:04d}-01-01"

    return f"{bucket}T00:00:00+00:00" if as_timestamp and "T" not in bucket else bucket


def aggregate_rows(
    rows,
    interval,
    time_column,
    group_columns=(),
    first_columns=(),
    max_columns=(),
    min_columns=(),
    last_columns=(),
    sum_columns=(),
    count_column="",
    timestamp_output=False,
):
    validate_time_interval(interval)
    if interval == "1D":
        return rows

    def number(value):
        if value == "" or value is None:
            return 0.0
        return float(value)

    grouped = {}
    order = []
    for row in sorted(rows, key=lambda item: str(item[time_column])):
        bucket = bucket_value(row[time_column], interval, as_timestamp=timestamp_output)
        key = tuple(row.get(column, "") for column in group_columns) + (bucket,)

        if key not in grouped:
            grouped[key] = {column: row.get(column, "") for column in group_columns}
            grouped[key][time_column] = bucket
            for column in first_columns:
                grouped[key][column] = row.get(column, "")
            for column in max_columns + min_columns + last_columns:
                grouped[key][column] = row.get(column, "")
            for column in sum_columns:
                grouped[key][column] = number(row.get(column, 0))
            if count_column:
                grouped[key][count_column] = 0
            order.append(key)
        else:
            for column in max_columns:
                if number(row.get(column, 0)) > number(grouped[key].get(column, 0)):
                    grouped[key][column] = row.get(column, "")
            for column in min_columns:
                if number(row.get(column, 0)) < number(grouped[key].get(column, 0)):
                    grouped[key][column] = row.get(column, "")
            for column in last_columns:
                grouped[key][column] = row.get(column, "")
            for column in sum_columns:
                grouped[key][column] = number(grouped[key].get(column, 0)) + number(row.get(column, 0))

        if count_column:
            grouped[key][count_column] += 1

    return [grouped[key] for key in order]

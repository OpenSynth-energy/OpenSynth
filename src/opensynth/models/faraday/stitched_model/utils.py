from calendar import monthrange
from datetime import date

import polars as pl


def sample_number_is_sufficient(
    df: pl.DataFrame,
    n_samples: int,
    year: int,
    month: int,
    sampled_features: pl.DataFrame | None = None,
):
    """Check if enough samples are generated to create a full month.

    Args:
        df (DataFrame): DataFrame with samples.
        n_samples (int): Number of samples to generate.
        year (int): Year.
        month (int): Month.
        sampled_features (DataFrame | None, optional): DataFrame with one
            column per feature and an additional column `n_required`, which
            specifies the number of samples to generate for that combination
            of features.
    Returns:
        bool
    """
    # Number of days in this specific month
    n_days = monthrange(year, month)[1]

    if sampled_features is None:
        # No check on features. All days should have enough samples and every
        # day in the month should be represented.
        return (
            df.group_by("date").len().min()["len"][0] >= n_samples
            and len(df["date"].unique()) >= n_days
        )
    else:
        # Checks if sufficient samples are present for specific combibations
        # of features for every day that generated.
        features = sampled_features.select(pl.exclude("n_required")).columns
        n_days_per_feature = (
            df.group_by("date", *features)
            .len()
            .join(sampled_features, on=features)
            .filter(pl.col("len") >= pl.col("n_required"))
            .group_by("date")
            .sum()
            .filter(pl.col("n_required") == n_samples)
            .shape
        )

        return n_days_per_feature[0] == n_days


def date_per_weekday_and_month(year: int) -> dict[int, dict[int, list[date]]]:
    """Return dictionary of dates in a year.

    This will create a nested dictionary with day of week as first key, and
    month as the second key.

    Args:
        year (int): Year.

    Returns:
        Dictionary with all days of the year as values and day_of_week and
            month as keys.
    """
    sample_df = (
        pl.date_range(date(year, 1, 1), date(year, 12, 31), "1d", eager=True)
        .alias("datetime")
        .to_frame()
        .with_columns(
            pl.col("datetime").dt.weekday().alias("weekday"),
            pl.col("datetime").dt.month().alias("month"),
        )
    )
    dates: dict[int, dict[int, list[date]]] = {
        i: {j: [] for j in range(1, 13)} for i in range(7)
    }
    for row in sample_df.iter_rows():
        dates[row[1] - 1][row[2]].append(row[0])

    return dates

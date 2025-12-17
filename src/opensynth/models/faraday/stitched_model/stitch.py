# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import polars as pl


def stitch_by_date(df: pl.DataFrame, n_samples: int) -> pl.DataFrame:
    """Sample an equal number of days per day (date) in the DataFrame.

    The sample number will be added as an additional column.

    Args:
        df (DataFrame): DataFrame with daily samples, where the
            distribution over days, as specified by the date column,
            can be different,
        n_samples (int): Number of samples to select per date.

    Returns:
        DataFrame with n_samples daily samples per day.
    """
    result = pl.concat(
        [p.sample(n_samples).with_row_index() for p in df.partition_by("date")]
    )
    return result


def stitch_by_date_and_features(
    df: pl.DataFrame, sampled_features: pl.DataFrame
) -> pl.DataFrame:
    """Sample an equal number of days per day (date) in the DataFrame.

    The sample number will be added as an additional column. The features
    of all samples with the same sample number will be consistent between
    days.

    Args:
        df (DataFrame): DataFrame with daily samples, where the
            distribution over days, as specified by the date column,
            can be different,
        sampled_features (DataFrame): DataFrame with sampled features.

    Returns:
        DataFrame with an equal number of samples per day.
    """
    features = sampled_features.select(pl.exclude("n_required")).columns
    result = pl.DataFrame()
    c = 0
    for feature_df in df.sort(features).partition_by(features):
        required_for_features = (
            feature_df.select(features)
            .head(1)
            .join(sampled_features, on=features)
        )
        if required_for_features.shape[0] == 0:
            continue
        n_required = required_for_features["n_required"][0]
        result = pl.concat(
            (
                result,
                pl.concat(
                    [
                        p.sample(n_required).with_row_index()
                        for p in feature_df.partition_by("date")
                    ],
                    how="vertical",
                ).with_columns(pl.col("index") + c),
            )
        )
        c += n_required

    return result


def stitch_samples(
    df: pl.DataFrame, sampled_features: pl.DataFrame | None, n_samples: int
) -> pl.DataFrame:
    """Select an equal number of samples per day.

    Args:
        df (DataFrame): DataFrame with generated daily samples.
        sampled_features (DataFrame, optional): Samples features, which will
            be used to select samples with consisten features over all days.
        n_samples (int): Number of samples to select. Will only be used
            if sampled_features is None.

    Returns:
        DataFrame with an equal number of samples per day.
    """
    result = (
        stitch_by_date(df, n_samples)
        if sampled_features is None
        else stitch_by_date_and_features(df, sampled_features)
    )
    return result

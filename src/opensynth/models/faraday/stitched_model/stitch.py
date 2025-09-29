import polars as pl


def stitch_by_date(df, n_samples):
    result = pl.concat(
        [p.sample(n_samples).with_row_index() for p in df.partition_by("date")]
    )
    return result


def stitch_by_date_and_features(df, sampled_features):
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


def stitch_samples(df, sampled_features, n_samples):
    result = (
        stitch_by_date(df, n_samples)
        if sampled_features is None
        else stitch_by_date_and_features(df, sampled_features)
    )
    return result

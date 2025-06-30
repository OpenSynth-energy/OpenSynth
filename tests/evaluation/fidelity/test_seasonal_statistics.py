
from opensynth.evaluation.fidelity.seasonal_statistics import (add_season,
seasonal_peaks,
calculate_seasonal_peaks,
print_seasonal_stats,
plot_seasonal_stats,
pairwise_seasonal_kstest,
)
import polars as pl
import pytest
import pandas as pd
import numpy as np

def test_add_season():
    df = pl.DataFrame({'datetime':['01-01-2021 11:00', '20-03-2022 21:00', '21-03-2022 07:00', "22-06-2011 09:00", "22-09-2027 12:00"]}).with_columns(pl.col('datetime').str.to_datetime())
    assert list(add_season(df)['season']) == ['winter', 'winter', 'spring', 'summer', 'fall']


@pytest.fixture
def peaky_profile():
    df = pd.DataFrame({
        'datetime': pd.date_range(start="2023-01-01 00:00", end="2023-12-31 23:45", freq="15min"),
        
    })
    df['value'] = np.random.random(df.shape[0])
 
    # 100 winter peaks
    idx = df[df['datetime'].dt.month == 1].sample(100).index
    df.loc[idx[:50], "value"] = 100
    df.loc[idx[50:], "value"] = 200

    # 1000 fall peaks
    idx = df[df['datetime'].dt.month == 10].sample(1000).index
    df.loc[idx[:500], "value"] = 100
    df.loc[idx[500:], "value"] = 200
    
    # other peaks
    idx = df.loc[add_season(df)["season"].isin(["spring", "summer"])].sample(6800).index
    df.loc[idx[:3400], "value"] = 100
    df.loc[idx[3400:], "value"] = 200
 
    return df

@pytest.fixture
def peaky_summer_profiles():
    df = pd.DataFrame({
        'datetime': pd.date_range(start="2023-01-01 00:00", end="2023-12-31 23:45", freq="15min"),
        
    })
    for profile in range(50):
        df[f'profile_{profile}'] = np.random.random(df.shape[0])
 
        # 100 summer peaks
        idx = df[add_season(df)['season'] == "summer"].sample(1000).index
        df.loc[idx, f'profile_{profile}'] += 100
 
    return df

@pytest.fixture
def peaky_winter_profiles():
    df = pd.DataFrame({
        'datetime': pd.date_range(start="2023-01-01 00:00", end="2023-12-31 23:45", freq="15min"),
        
    })
    for profile in range(50):
        df[f'profile_{profile}'] = np.random.random(df.shape[0])
 
        # 100 summer peaks
        idx = df[add_season(df)['season'] == "winter"].sample(1000).index
        df.loc[idx, f'profile_{profile}'] += 100
 
    return df


@pytest.mark.parametrize(
    "quantile,expected",
    (
        (0.8, {"springsummer":6800, "fall":1000, "winter":100}),
        (0.9, {"springsummer":3400, "fall":500, "winter":50}),
    )
)
def test_seasonal_peaks(peaky_profile, quantile, expected):
    result = seasonal_peaks(peaky_profile, quantile=quantile).set_index("season").to_dict()["value"]
    assert result['fall'] == expected['fall']
    assert result['winter'] == expected['winter']
    assert (result['summer'] + result['spring']) == expected['springsummer']

def test_calculate_seasonal_peaks(peaky_profile):
    result = calculate_seasonal_peaks({"df":peaky_profile})
    assert result.shape == (4,4)
    result = result.set_index('season')
    assert result.loc['winter', 'value'] < result.loc['fall', 'value']
    assert result.loc['fall', 'value'] < result.loc['summer', 'value']

def test_print_seasonal_stats(capsys, peaky_profile):
    result = calculate_seasonal_peaks({"name":peaky_profile})
    print_seasonal_stats(result)
    captured = capsys.readouterr()
    assert "shape" in captured.out
    assert "name" in captured.out

def test_print_plot_stats(peaky_profile):
    """Only tests if plot runs without errors."""
    result = calculate_seasonal_peaks({"df":peaky_profile})
    plot_seasonal_stats(result)



def test_pairwise_seasonal_kstest(peaky_summer_profiles, peaky_winter_profiles):
    result = calculate_seasonal_peaks({"summer":peaky_summer_profiles, "winter": peaky_winter_profiles})
    stats_result = pairwise_seasonal_kstest(result, 'winter', 'summer')
    assert stats_result.set_index("season").loc['summer']['p_value'] < 1e-10
    assert stats_result.set_index("season").loc['winter']['p_value'] < 1e-10
    assert stats_result.set_index("season").loc['spring']['p_value'] > 0.05
    assert stats_result.set_index("season").loc['fall']['p_value'] > 0.05
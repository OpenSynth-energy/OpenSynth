# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
import random

import pandas as pd

from opensynth.datasets.low_carbon_london import split_households


class TestSplitHouseholdIds:

    df = pd.DataFrame({"ID": [f"H{i}" for i in range(40)]})

    def test_fraction_respected(self):
        train, holdout = split_households.split_household_ids(
            self.df, "ID", sample_fraction=0.75, seed=1
        )
        assert len(train) == 30 and len(holdout) == 10
        assert set(train) | set(holdout) == set(self.df["ID"])

    def test_seed_makes_split_independent_of_global_stream(self):
        first = split_households.split_household_ids(self.df, "ID", seed=7)
        # Disturb the global random stream between calls
        random.random()
        second = split_households.split_household_ids(self.df, "ID", seed=7)
        assert first == second

    def test_seed_42_matches_historical_global_behaviour(self):
        # preprocess_ne passes config.SAMPLING_SEED (42); this must
        # reproduce the original module-level random.seed(42) split
        # so existing datasets keep their membership
        random.seed(42)
        ids = self.df["ID"].unique().tolist()
        random.shuffle(ids)
        expected = ids[:30]
        train, _ = split_households.split_household_ids(self.df, "ID", seed=42)
        assert train == expected

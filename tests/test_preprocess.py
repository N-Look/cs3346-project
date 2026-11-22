"""Regression checks for in-memory feature cleaning and source row alignment."""

import numpy as np
import pandas as pd

from src import preprocess


def test_cleaning_reads_current_inputs_in_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "DAYS", ("Monday",))
    source = tmp_path / "Monday.parquet"
    original = pd.DataFrame({
        "day": ["Monday", "Monday"],
        "Timestamp": pd.to_datetime(["2017-07-03 15:00:02", "2017-07-03 15:00:01"]),
        "attack_type": ["BENIGN", "Bot"],
        "is_attack": [False, True],
        "is_attempted": [False, False],
        "Src IP": ["source-a", "source-b"],
        "Average Packet Size": [72.0, 48.0],
        "Packets": [-0.0, 2.0],
        **{name: [-1.0, np.inf] for name in preprocess.NONNEGATIVE_FEATURES},
    })
    original.to_parquet(source)

    metadata, features, columns = preprocess.load_experiment_inputs(tmp_path)
    assert list(tmp_path.iterdir()) == [source]
    assert columns == ["Packets", *preprocess.NONNEGATIVE_FEATURES]
    assert metadata["attack_type"].tolist() == ["BENIGN", "Bot"]
    np.testing.assert_array_equal(features[:, 0], [0.0, 2.0])
    assert not np.signbit(features[0, 0])
    assert np.isnan(features[:, 1:]).all()

    original.loc[1, "Packets"] = 99
    replacement = tmp_path / "replacement.parquet"
    original.to_parquet(replacement)
    replacement.replace(source)
    _, updated, _ = preprocess.load_experiment_inputs(tmp_path)

    assert updated[1, 0] == 99

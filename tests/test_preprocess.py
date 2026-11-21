"""Regression checks for saved feature cleaning and source row alignment."""

import numpy as np
import pandas as pd

from src import preprocess


def test_saved_features_are_reused_and_rebuilt_with_source_changes(tmp_path, monkeypatch):
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

    (saved,) = preprocess.prepare_cleaned_files(tmp_path)
    first_stat = saved.stat()
    metadata, features, columns = preprocess.load_experiment_inputs(tmp_path)

    assert saved.stat().st_ino == first_stat.st_ino
    assert saved.stat().st_mtime_ns == first_stat.st_mtime_ns
    assert columns == ["Packets", *preprocess.NONNEGATIVE_FEATURES]
    assert list(pd.read_parquet(saved)) == columns
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

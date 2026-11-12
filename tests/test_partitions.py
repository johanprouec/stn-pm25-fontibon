"""Chronological partitioning: sizes match the Primera Entrega report and
partitions are strictly ordered in time with no overlap."""
from src.data.fontibon import chronological_cuts


def test_partition_sizes_match_report(fontibon_features):
    cortes = chronological_cuts(len(fontibon_features))
    assert cortes["train"] == (0, int(len(fontibon_features) * 0.70))
    assert cortes["val"][0] == cortes["train"][1]
    assert cortes["test"][0] == cortes["val"][1]
    assert cortes["test"][1] == len(fontibon_features)


def test_partition_windows_match_report_counts(fontibon_partitions):
    # Reported in Primer Entrega DL.pdf, Tabla 2, and cross-verified against
    # the notebook's own printed output (see docs/PHASE1_AUDIT.md).
    expected = {"train": 10802, "val": 2564, "test": 2574}
    for name, n in expected.items():
        _, y, _ = fontibon_partitions[name]
        assert len(y) == n, f"{name}: expected {n} windows, got {len(y)}"


def test_partitions_are_chronologically_ordered(fontibon_partitions):
    _, _, dates_train = fontibon_partitions["train"]
    _, _, dates_val = fontibon_partitions["val"]
    _, _, dates_test = fontibon_partitions["test"]

    assert dates_train.max() < dates_val.min()
    assert dates_val.max() < dates_test.min()


def test_no_timestamp_overlap_between_partitions(fontibon_partitions):
    dates_train = set(fontibon_partitions["train"][2])
    dates_val = set(fontibon_partitions["val"][2])
    dates_test = set(fontibon_partitions["test"][2])

    assert dates_train.isdisjoint(dates_val)
    assert dates_val.isdisjoint(dates_test)
    assert dates_train.isdisjoint(dates_test)

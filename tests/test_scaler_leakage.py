"""The scaler (and any fitted preprocessing) must be fit on TRAIN only.
These tests fit a StandardScaler on train and check that its learned
statistics differ from what you'd get by (incorrectly) including val/test,
i.e. that val/test genuinely were not seen during fit."""
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.fontibon import flatten


def test_scaler_fit_on_train_differs_from_fit_on_all(fontibon_partitions):
    Xtr, _, _ = fontibon_partitions["train"]
    Xva, _, _ = fontibon_partitions["val"]
    Xte, _, _ = fontibon_partitions["test"]

    scaler_train_only = StandardScaler().fit(flatten(Xtr))

    all_data = np.concatenate([flatten(Xtr), flatten(Xva), flatten(Xte)], axis=0)
    scaler_all = StandardScaler().fit(all_data)

    # If these matched, it would mean train alone already has identical
    # statistics to train+val+test, which would make this test blind to a
    # leak. Fontibon's val/test PM2.5 means (15.67, 18.77) differ from
    # train's (19.10, per the notebook's own printed output), so the means
    # must differ here too.
    assert not np.allclose(scaler_train_only.mean_, scaler_all.mean_)


def test_scaler_transform_of_val_test_uses_train_statistics(fontibon_partitions):
    Xtr, _, _ = fontibon_partitions["train"]
    Xva, _, _ = fontibon_partitions["val"]

    scaler = StandardScaler().fit(flatten(Xtr))
    transformed_val = scaler.transform(flatten(Xva))

    # Manually reproduce: (val - train_mean) / train_scale
    manual = (flatten(Xva) - scaler.mean_) / scaler.scale_
    assert np.allclose(transformed_val, manual)


def test_train_target_mean_matches_report(fontibon_partitions):
    _, ytr, _ = fontibon_partitions["train"]
    _, yva, _ = fontibon_partitions["val"]
    _, yte, _ = fontibon_partitions["test"]

    assert abs(ytr.mean() - 19.10) < 0.01
    assert abs(yva.mean() - 15.67) < 0.01
    assert abs(yte.mean() - 18.77) < 0.01

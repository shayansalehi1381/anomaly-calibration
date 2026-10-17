"""Leakage tests: the memory bank, held-out normal pool and anomaly set must be disjoint."""

import dataclasses

import numpy as np
import pytest

from anomcal.data.splits import (
    ClassSplit,
    LeakageError,
    assert_disjoint,
    make_calib_split,
    make_official_split,
    make_split,
    n_memory,
    partition_heldout,
)
from anomcal.data.visa import EXPECTED_1CLS_COUNTS, VisaRecord, records_from_counts

REAL_SIZED = records_from_counts(EXPECTED_1CLS_COUNTS)


@pytest.mark.parametrize("cls", sorted(EXPECTED_1CLS_COUNTS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_calib_split_is_disjoint_and_complete_for_every_class(cls, seed):
    split = make_calib_split(REAL_SIZED, cls, seed, memory_fraction=0.5)
    assert_disjoint(split, REAL_SIZED)
    n_tr, n_te, n_an = EXPECTED_1CLS_COUNTS[cls]
    assert len(split.memory) == n_memory(n_tr, 0.5)
    # every normal image is in exactly one of memory / held-out
    assert len(split.memory) + len(split.heldout_normal) == n_tr + n_te
    assert len(split.anomaly) == n_an
    by_id = {r.image_id: r for r in REAL_SIZED}
    assert all(by_id[i].official_split == "train" for i in split.memory)
    # all official test normals are held out, never in the memory bank
    test_normals = {r.image_id for r in REAL_SIZED if r.cls == cls and r.official_split == "test"}
    test_normals = {i for i in test_normals if by_id[i].label == 0}
    assert test_normals <= set(split.heldout_normal)


@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_disjoint_for_many_fractions_and_seeds(fraction):
    for seed in range(20):
        s = make_calib_split(REAL_SIZED, "cashew", seed, fraction)
        assert not set(s.memory) & set(s.heldout_normal)
        assert not set(s.memory) & set(s.anomaly)
        assert not set(s.heldout_normal) & set(s.anomaly)


def test_split_is_deterministic_and_seed_dependent():
    a = make_calib_split(REAL_SIZED, "candle", 0)
    assert a == make_calib_split(REAL_SIZED, "candle", 0)
    assert set(a.memory) != set(make_calib_split(REAL_SIZED, "candle", 1).memory)


def test_split_does_not_depend_on_csv_row_order_or_other_classes():
    a = make_calib_split(REAL_SIZED, "pcb1", 3)
    shuffled = list(REAL_SIZED)
    np.random.default_rng(0).shuffle(shuffled)
    only_pcb1 = [r for r in shuffled if r.cls == "pcb1"]
    assert make_calib_split(only_pcb1, "pcb1", 3) == a


def test_official_split_uses_all_train_normals():
    s = make_official_split(REAL_SIZED, "candle")
    assert (len(s.memory), len(s.heldout_normal), len(s.anomaly)) == (900, 100, 100)
    assert_disjoint(s, REAL_SIZED)
    assert make_split(REAL_SIZED, "candle", "official", 0) == s


def test_unknown_protocol():
    with pytest.raises(ValueError):
        make_split(REAL_SIZED, "candle", "nope", 0)


# ---- the guard must FAIL on any overlap -------------------------------------------


def _base() -> ClassSplit:
    return make_calib_split(REAL_SIZED, "fryum", 0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: dataclasses.replace(s, heldout_normal=s.heldout_normal + (s.memory[0],)),
        lambda s: dataclasses.replace(s, anomaly=s.anomaly + (s.memory[0],)),
        lambda s: dataclasses.replace(s, anomaly=s.anomaly + (s.heldout_normal[0],)),
        lambda s: dataclasses.replace(s, memory=s.memory + (s.memory[0],)),
    ],
    ids=["memory&heldout", "memory&anomaly", "heldout&anomaly", "duplicate-in-memory"],
)
def test_overlap_raises(mutate):
    with pytest.raises(LeakageError):
        assert_disjoint(mutate(_base()))


def test_test_normal_in_memory_bank_raises():
    s = _base()
    test_normal = next(
        r.image_id
        for r in REAL_SIZED
        if r.cls == "fryum" and r.official_split == "test" and r.label == 0
    )
    bad = dataclasses.replace(
        s,
        memory=s.memory + (test_normal,),
        heldout_normal=tuple(i for i in s.heldout_normal if i != test_normal),
    )
    assert_disjoint(bad)  # ids alone are still disjoint...
    with pytest.raises(LeakageError, match="non-train"):
        assert_disjoint(bad, REAL_SIZED)  # ...but the record check catches it


def test_anomaly_in_heldout_pool_raises():
    s = _base()
    bad = dataclasses.replace(
        s, heldout_normal=s.heldout_normal + (s.anomaly[0],), anomaly=s.anomaly[1:]
    )
    with pytest.raises(LeakageError, match="anomaly inside"):
        assert_disjoint(bad, REAL_SIZED)


def test_same_file_under_two_ids_raises(tmp_path):
    """Two ids pointing to the same file on disk (one in memory, one held out)."""
    rel = "candle/Data/Images/Normal/0001.JPG"
    a = VisaRecord("candle/normal/0001", "candle", "train", 0, rel, None)
    b = VisaRecord("candle/normal/0001-copy", "candle", "train", 0, rel, None)
    s = ClassSplit("candle", "calib", 0, 0.5, (a.image_id,), (b.image_id,), ())
    assert_disjoint(s, [a, b])  # ids differ
    with pytest.raises(LeakageError, match="both"):
        assert_disjoint(s, [a, b], root=tmp_path)


def test_unknown_id_raises():
    s = dataclasses.replace(_base(), anomaly=("fryum/anomaly/does-not-exist",))
    with pytest.raises(LeakageError, match="not in the split file"):
        assert_disjoint(s, REAL_SIZED)


def test_fake_dataset_split_checks_file_paths(fake_visa_root, fake_records):
    s = make_calib_split(fake_records, "candle", 0)
    assert_disjoint(s, fake_records, fake_visa_root)


# ---- helpers -----------------------------------------------------------------------


def test_n_memory_bounds():
    assert n_memory(450, 0.5) == 225
    assert n_memory(2, 0.01) == 1 and n_memory(2, 0.99) == 1
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            n_memory(10, bad)
    with pytest.raises(ValueError):
        n_memory(1, 0.5)


def test_limited_keeps_disjointness_and_order():
    s = _base().limited(5)
    assert len(s.memory) == len(s.heldout_normal) == len(s.anomaly) == 5
    assert s.memory == _base().memory[:5] and s.max_images == 5
    assert_disjoint(s, REAL_SIZED)
    assert _base().limited(None) == _base()
    with pytest.raises(ValueError):
        _base().limited(0)


def test_partition_heldout_is_disjoint_and_complete():
    ids = [f"x/{i}" for i in range(101)]
    cal, test = partition_heldout(ids, 0.5, np.random.default_rng(0))
    assert not set(cal) & set(test) and set(cal) | set(test) == set(ids)
    assert len(cal) == 50 or len(cal) == 51
    cal2, _ = partition_heldout(ids, 0.5, np.random.default_rng(0))
    assert cal == cal2
    with pytest.raises(ValueError):
        partition_heldout(ids, 1.0, np.random.default_rng(0))


def test_to_dict_roundtrip_fields():
    d = _base().to_dict()
    assert d["class"] == "fryum" and d["protocol"] == "calib" and d["memory_fraction"] == 0.5
    assert len(d["memory"]) == 225

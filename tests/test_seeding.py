import random
import subprocess
import sys

import numpy as np
import pytest

from anomcal.seeding import derive_seed, item_rng, set_global_seed


def test_derive_seed_is_deterministic():
    assert derive_seed("candle/0001", "blur", 1) == derive_seed("candle/0001", "blur", 1)


def test_derive_seed_distinguishes_parts():
    seeds = {
        derive_seed("candle/0001", "blur", 1),
        derive_seed("candle/0001", "blur", 2),
        derive_seed("candle/0002", "blur", 1),
        derive_seed("candle/0001", "noise", 1),
    }
    assert len(seeds) == 4


def test_derive_seed_separator_prevents_concatenation_collisions():
    assert derive_seed("ab", "c") != derive_seed("a", "bc")


def test_derive_seed_distinguishes_types():
    # 1 (int) and "1" (str) are different identities
    assert derive_seed(1) != derive_seed("1")


def test_derive_seed_range():
    s = derive_seed("x")
    assert 0 <= s < 2**63


def test_derive_seed_requires_parts():
    with pytest.raises(ValueError):
        derive_seed()


def test_derive_seed_is_stable_across_processes():
    # Python's hash() would fail this test: it is salted per process.
    code = "from anomcal.seeding import derive_seed; print(derive_seed('candle/0001', 'jpeg', 3))"
    out = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert out == {str(derive_seed("candle/0001", "jpeg", 3))}


def test_item_rng_reproducible_and_order_independent():
    a1 = item_rng("img_a").normal(size=3)
    item_rng("img_b").normal(size=100)  # drawing for another item in between...
    a2 = item_rng("img_a").normal(size=3)  # ...does not change img_a's numbers
    np.testing.assert_array_equal(a1, a2)


def test_set_global_seed_fixes_python_and_numpy():
    set_global_seed(123)
    first = (random.random(), np.random.rand())
    set_global_seed(123)
    assert (random.random(), np.random.rand()) == first


def test_set_global_seed_accepts_large_seed():
    set_global_seed(2**40)  # numpy legacy RNG needs < 2**32; we reduce it internally


@pytest.mark.parametrize("bad", [-1, 1.5, "3"])
def test_set_global_seed_rejects_bad_seed(bad):
    with pytest.raises(ValueError):
        set_global_seed(bad)

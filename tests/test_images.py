import numpy as np
from PIL import Image

from anomcal.data.images import load_image, load_images, load_mask


def test_load_image_shape_dtype(fake_visa_root, fake_records):
    img = load_image(fake_records[0].image_path(fake_visa_root), size=32)
    assert img.shape == (32, 32, 3) and img.dtype == np.uint8


def test_grayscale_is_converted_to_rgb(tmp_path):
    p = tmp_path / "g.png"
    Image.fromarray(np.full((10, 20), 100, np.uint8)).save(p)
    img = load_image(p, size=8)
    assert img.shape == (8, 8, 3) and (img == 100).all()


def test_mask_is_binarised_and_stays_binary(tmp_path):
    m = np.zeros((40, 60), np.uint8)
    m[10:20, 10:30] = 7  # class-id coded defect
    p = tmp_path / "m.png"
    Image.fromarray(m).save(p)
    mask = load_mask(p, size=20)
    assert mask.dtype == bool and mask.any() and not mask.all()
    assert load_mask(None, size=20).sum() == 0


def test_threaded_loading_equals_sequential(fake_visa_root, fake_records):
    paths = [r.image_path(fake_visa_root) for r in fake_records[:10]]
    np.testing.assert_array_equal(load_images(paths, 32, 0), load_images(paths, 32, 3))

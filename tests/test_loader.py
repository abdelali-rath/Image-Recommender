import os
import sys
import pytest
from PIL import Image

# Ensure project root is on path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_recommender.data.loader import (
    load_image, preprocess_image, load_images_generator, generate_image_id
)


def create_dummy_image(path, color=(255, 0, 0)):
    img = Image.new('RGB', (10, 10), color)
    img.save(path)


def test_load_image_and_preprocess(tmp_path):
    # Create dummy image file
    img_path = tmp_path / "test.jpg"
    create_dummy_image(img_path)

    img = load_image(str(img_path))
    assert img is not None and img.mode == 'RGB'

    resized = preprocess_image(img, size=(5, 5))
    assert resized.size == (5, 5)

    # Loading a non-image path returns None
    assert load_image(str(tmp_path / "no_image.jpg")) is None


def test_load_images_generator(tmp_path):
    # Setup directory with various files
    root = tmp_path / "dataset"
    sub = root / "subdir"
    sub.mkdir(parents=True)
    valid = [root / "a.JPG", sub / "b.png"]
    invalid = [root / "c.txt"]
    for f in valid:
        create_dummy_image(f)
    for f in invalid:
        f.write_text("not an image")

    paths = list(load_images_generator(str(root)))
    # Should only yield valid extensions, case-insensitive
    assert set(paths) == set(str(p) for p in valid)


def test_generate_image_id_consistency():
    p = '/some/path/image.jpg'
    id1 = generate_image_id(p)
    id2 = generate_image_id(p)
    assert id1 == id2 and len(id1) == 64  # SHA 256 hex length
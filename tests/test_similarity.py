import numpy as np
import pytest
from PIL import Image

from image_recommender.similarity.hist_similarity import (
    compute_histogram, image_color_similarity
)
from image_recommender.similarity.similarity_phash import (
    compute_phash, phash_similarity
)
from image_recommender.similarity.similarity_embedding import (
    build_annoy_index, load_annoy_index, EMBEDDING_DIM
)


def create_solid_image(color, size=(16, 16)):
    return Image.new('RGB', size, color)


def test_compute_histogram_and_color():
    img_black = create_solid_image((0, 0, 0))
    img_white = create_solid_image((255, 255, 255))

    hist = compute_histogram(img_black, bins=4)
    assert hist.shape == (12,)
    # Identical images -> zero distance
    assert image_color_similarity(img_black, img_black, bins=4) == pytest.approx(0.0, abs=1e-6)
    # Black vs white yields >0 distance
    assert image_color_similarity(img_black, img_white, bins=4) > 0


def test_phash_and_similarity():
    img1 = create_solid_image((128, 128, 128))
    img2 = create_solid_image((128, 128, 128))
    img3 = create_solid_image((0, 0, 0))

    # pHash identical images
    assert compute_phash(img1) - compute_phash(img2) == 0
    assert phash_similarity(img1, img2) == 0
    # Different images should have non-negative distance
    assert phash_similarity(img1, img3) >= 0


def test_build_and_load_annoy_index(tmp_path):
    # Create dummy zero embeddings
    embeddings = {i: [0.0] * EMBEDDING_DIM for i in range(5)}
    idx_file = tmp_path / "index.ann"

    build_annoy_index(embeddings, str(idx_file), n_trees=5)
    assert idx_file.exists()

    idx = load_annoy_index(str(idx_file))
    # Query zero vector for top-3 neighbors
    nns = idx.get_nns_by_vector([0.0] * EMBEDDING_DIM, 3)
    assert len(nns) == 3

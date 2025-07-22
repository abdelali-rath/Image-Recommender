import os
import json
import sys
from PIL import Image

# Allow local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_recommender.similarity_embedding import compute_clip_embedding, load_annoy_index, EMBEDDING_DIM
from image_recommender.database import get_image_by_id
from image_recommender.loader import load_image, preprocess_image


def load_index_and_mapping(index_path: str, mapping_path: str):
    """
    Loads the Annoy index and mapping file.
    Returns (annoy_index, dict[int → image_id])
    """
    index = load_annoy_index(index_path)
    with open(mapping_path, "r") as f:
        raw_map = json.load(f)
    mapping = {int(k): v for k, v in raw_map.items()}
    return index, mapping

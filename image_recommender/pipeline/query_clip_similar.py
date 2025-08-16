import os
import json
import sys
from PIL import Image

# Allow local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from similarity.similarity_embedding import (
    compute_clip_embedding,
    load_annoy_index,
    EMBEDDING_DIM,
)
from data.database import get_image_by_id
from data.loader import load_image, preprocess_image


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


def find_top_k_similar(image_path: str, index_path: str, mapping_path: str, k: int = 5):
    """
    Computes CLIP embedding for input image and finds k most similar images.
    Prints image IDs and paths.
    """
    image = load_image(image_path)
    if image is None:
        print("❌ Could not load input image.")
        return

    image = preprocess_image(image)
    embedding = compute_clip_embedding(image)

    index, id_map = load_index_and_mapping(index_path, mapping_path)

    nearest_idxs, distances = index.get_nns_by_vector(
        embedding.tolist(), k, include_distances=True
    )

    print(f"\n🔍 Top-{k} similar images to {image_path}:\n")
    for rank, (i, dist) in enumerate(zip(nearest_idxs, distances), 1):
        image_id = id_map.get(i, "<unknown>")
        db_result = get_image_by_id(image_id)
        if db_result:
            path, width, height = db_result
            print(f"{rank}. 🖼️ {path}  (distance: {dist:.4f})")
        else:
            print(f"{rank}. 🆔 {image_id}  (distance: {dist:.4f}) – Not found in DB")

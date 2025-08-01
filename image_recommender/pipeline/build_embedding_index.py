import os
import json
from tqdm import tqdm
from PIL import Image
import sys

# Add project root to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.database import connect_db
from data.loader import load_image, preprocess_image
from similarity.similarity_embedding import compute_clip_embedding, build_annoy_index, EMBEDDING_DIM

# Define base project directory (2 levels up from this file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths for output files
index_out = os.path.join(BASE_DIR, "data", "out", "clip_index.ann")
mapping_out = os.path.join(BASE_DIR, "data", "out", "index_to_id.json")


def get_all_images_from_db():
    """
    Returns a list of (image_id, path) from the SQLite database.
    """
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, path FROM images;")
        return cursor.fetchall()


def build_and_save_embeddings(index_path: str, mapping_path: str, max_images=None):
    """
    Loads images from DB, computes CLIP embeddings, builds Annoy index.

    Args:
        index_path (str): File path to save Annoy index
        mapping_path (str): File path to save ID mapping (index → image_id)
        max_images (int): Maximum number of images to process
    """
    data = get_all_images_from_db()
    if max_images:
        data = data[:max_images]

    print(f"🧠 Processing {len(data)} images (CLIP Embeddings)...")

    from annoy import AnnoyIndex
    index = AnnoyIndex(EMBEDDING_DIM, metric="angular")
    mapping = {}

    for i, (image_id, path) in enumerate(tqdm(data, desc="Embedding images")):
        img = load_image(path)
        if img is None:
            continue
        img = preprocess_image(img)

        embedding = compute_clip_embedding(img)
        index.add_item(i, embedding.tolist())
        mapping[i] = image_id

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)

    index.build(10)
    index.save(index_path)

    with open(mapping_path, "w") as f:
        json.dump(mapping, f)

    print(f"✅ Saved Annoy index to {index_path}")
    print(f"✅ Saved index-to-ID mapping to {mapping_path}")


if __name__ == "__main__":
    build_and_save_embeddings(index_out, mapping_out)

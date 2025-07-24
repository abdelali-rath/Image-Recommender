import json
import os
from collections import defaultdict

from PIL import Image
import numpy as np

from image_recommender.loader import load_image, preprocess_image
from image_recommender.similarity_embedding import compute_clip_embedding, load_annoy_index
from image_recommender.hist_similarity import image_color_similarity
from image_recommender.similarity_phash import phash_similarity
from image_recommender.database import get_image_by_id

# Weights for combining scores (adjust as needed)
WEIGHTS = {
    "clip": 0.5,
    "color": 0.3,
    "phash": 0.2
}


def load_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def combined_similarity_search(
    input_path,  # str or list of str
    clip_index_path: str,
    clip_mapping_path: str,
    k_clip: int = 20,
    top_k_result: int = 5
):
    """
    Combines CLIP, histogram, and pHash similarities to find the best matches.
    Supports one or multiple input images.

    Returns: List of (path, combined_score)
    """
    # Handle single or multiple input images
    if isinstance(input_path, str):
        input_path = [input_path]

    input_images = []
    embeddings = []

    for path in input_path:
        img = load_image(path)
        if img is None:
            continue
        img = preprocess_image(img)
        input_images.append(img)
        embeddings.append(compute_clip_embedding(img))

    if not embeddings:
        print("❌ Could not load any input image.")
        return []

    # Average embedding vector
    input_embedding = sum(embeddings) / len(embeddings)

    # Load CLIP index and mapping
    clip_index = load_annoy_index(clip_index_path)
    index_to_id = load_mapping(clip_mapping_path)

    # Get top-k CLIP neighbors with distances
    clip_results, distances = clip_index.get_nns_by_vector(
        input_embedding.tolist(), k_clip, include_distances=True
    )

    scores = []

    for idx, clip_dist in zip(clip_results, distances):
        candidate_id = index_to_id[idx]
        db_entry = get_image_by_id(candidate_id)
        if not db_entry:
            continue

        path, width, height = db_entry
        candidate_img = load_image(path)
        if candidate_img is None:
            continue
        candidate_img = preprocess_image(candidate_img)

        # CLIP similarity
        clip_sim = 1.0 - (clip_dist / 2.0)  # angular [0,2] → similarity [1,0]

        # Average color and pHash similarity across all query images
        color_sims = []
        phash_sims = []

        for input_img in input_images:
            color_dist = image_color_similarity(input_img, candidate_img)
            color_sim = 1.0 / (1.0 + color_dist)

            phash_dist = phash_similarity(input_img, candidate_img)
            phash_sim = 1.0 / (1.0 + phash_dist)

            color_sims.append(color_sim)
            phash_sims.append(phash_sim)

        avg_color_sim = sum(color_sims) / len(color_sims)
        avg_phash_sim = sum(phash_sims) / len(phash_sims)

        # Combined score
        combined = (
            WEIGHTS["clip"] * clip_sim +
            WEIGHTS["color"] * avg_color_sim +
            WEIGHTS["phash"] * avg_phash_sim
        )

        scores.append((path, combined))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k_result]

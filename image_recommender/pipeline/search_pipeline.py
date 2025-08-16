import json
import os
from collections import defaultdict

from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from heapq import heappush, heappushpop, nlargest

from image_recommender.data.loader import load_image, preprocess_image
from image_recommender.similarity.similarity_embedding import (
    compute_clip_embedding,
    load_annoy_index,
)
from image_recommender.similarity.hist_similarity import image_color_similarity
from image_recommender.similarity.similarity_phash import phash_similarity
from image_recommender.data.database import get_image_by_id

# Weights for combining scores (adjust as needed)
WEIGHTS = {"clip": 0.5, "color": 0.3, "phash": 0.2}

# Early termination toggle + chunking (kept internal; no signature change)
_EARLY_TERMINATION = True
_CHUNK_MULTIPLIER = 4  # submit work in chunks so we can prune between chunks


def load_mapping(mapping_path):
    with open(mapping_path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def combined_similarity_search(
    input_path,  # str or list of str
    clip_index_path: str,
    clip_mapping_path: str,
    k_clip: int = 20,
    top_k_result: int = 5,
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

    # Prefetch candidate paths on main thread (avoid DB access in worker threads)
    candidates = []
    for idx, clip_dist in zip(clip_results, distances):
        candidate_id = index_to_id[idx]
        db_entry = get_image_by_id(candidate_id)
        if not db_entry:
            continue
        path, width, height = db_entry
        # Map Annoy angular distance to similarity (kept your existing mapping)
        clip_sim = 1.0 - (clip_dist / 2.0)
        candidates.append((path, clip_dist, clip_sim))

    # Sort by CLIP similarity desc so our upper bound shrinks monotonically
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Parallel re-ranking (color + pHash) per candidate
    def _score_candidate(path: str, clip_dist: float):
        candidate_img = load_image(path)
        if candidate_img is None:
            return None
        candidate_img = preprocess_image(candidate_img)

        # CLIP similarity
        clip_sim_local = 1.0 - (clip_dist / 2.0)  # angular [0,2] → similarity [1,0]

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
            WEIGHTS["clip"] * clip_sim_local
            + WEIGHTS["color"] * avg_color_sim
            + WEIGHTS["phash"] * avg_phash_sim
        )
        return (path, combined)

    scores_heap = []  # min-heap of (combined, path)
    if candidates:
        max_workers = min(multiprocessing.cpu_count(), len(candidates)) or 1
        chunk_size = max_workers * _CHUNK_MULTIPLIER

        i = 0
        while i < len(candidates):
            chunk = candidates[i : i + chunk_size]

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [
                    ex.submit(_score_candidate, path, dist)
                    for (path, dist, _sim) in chunk
                ]
                for fut in as_completed(futs):
                    res = fut.result()
                    if not res:
                        continue
                    path, combined = res
                    if len(scores_heap) < top_k_result:
                        heappush(scores_heap, (combined, path))
                    else:
                        heappushpop(scores_heap, (combined, path))

            i += len(chunk)

            # Early termination check (only if we already filled top-k)
            if (
                _EARLY_TERMINATION
                and len(scores_heap) >= top_k_result
                and i < len(candidates)
            ):
                # Upper bound for any remaining candidate:
                # assume color=1 and phash=1 (best possible), with next candidate's clip_sim.
                next_clip_sim = candidates[i][2]
                upper_bound = (
                    WEIGHTS["clip"] * next_clip_sim
                    + WEIGHTS["color"] * 1.0
                    + WEIGHTS["phash"] * 1.0
                )
                worst_in_topk = scores_heap[0][0]  # min in heap
                if upper_bound <= worst_in_topk:
                    break

    # Convert heap to sorted list desc
    top = nlargest(top_k_result, scores_heap)
    return [(path, combined) for (combined, path) in top]

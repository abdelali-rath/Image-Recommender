import time, statistics as stats
import numpy as np
from PIL import Image
import torch

from image_recommender.similarity.similarity_embedding import (
    compute_clip_embedding,
    compute_clip_embeddings_batch,
)


def make_dataset(n=32):
    # synthetic images of varying solid colors
    imgs = []
    for i in range(n):
        c = (i * 7 % 256, i * 13 % 256, i * 29 % 256)
        imgs.append(Image.new("RGB", (224, 224), c))
    return imgs


def describe(name, xs):
    print(
        f"{name}: mean={stats.mean(xs) * 1000:.2f} ms | p50={np.percentile(xs, 50) * 1000:.2f} ms | p95={np.percentile(xs, 95) * 1000:.2f} ms | n={len(xs)}"
    )


def main():
    imgs = make_dataset(64)

    # per-image
    t = []
    for img in imgs:
        t0 = time.perf_counter()
        _ = compute_clip_embedding(img)
        t.append(time.perf_counter() - t0)
    describe("Per-image encode (loop)", t)

    # one batch
    t0 = time.perf_counter()
    _ = compute_clip_embeddings_batch(imgs)
    t1 = time.perf_counter()
    print(f"One batch encode: {(t1 - t0) * 1000:.2f} ms for {len(imgs)} images")


if __name__ == "__main__":
    torch.set_num_threads(1)  # optional: reduce variance on CPU
    main()

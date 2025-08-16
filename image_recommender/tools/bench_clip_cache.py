import argparse, time, gc, statistics as stats
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import clip

# import your cached getter from your module
# adjust the import path if your file name differs
from image_recommender.similarity.similarity_embedding import get_clip_model

def _sync_if_cuda(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()

def time_clip_load(repeats: int, device: str, model_name: str = "ViT-B/32"):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        _sync_if_cuda(device)
        times.append(time.perf_counter() - t0)
        # cleanup
        del model, preprocess
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return times

def time_cached_getter(repeats: int, device: str):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        get_clip_model()
        _sync_if_cuda(device)
        times.append(time.perf_counter() - t0)
    return times

def compute_once(model, preprocess, image: Image.Image, device: str):
    img = preprocess(image).unsqueeze(0).to(device)
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        emb = model.encode_image(img)
    _sync_if_cuda(device)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return time.perf_counter() - t0, emb.squeeze().cpu().numpy().astype(np.float32)

def time_embedding(image_path: str, repeats: int, device: str):
    image = Image.open(image_path).convert("RGB")

    # (A) new model for every embedding (simulates not using a cache)
    cold_times = []
    for _ in range(repeats):
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        dt, _ = compute_once(model, preprocess, image, device)
        cold_times.append(dt)
        del model, preprocess
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # (B) one cached model reused
    model, preprocess = get_clip_model()
    warm_times = []
    for _ in range(repeats):
        dt, _ = compute_once(model, preprocess, image, device)
        warm_times.append(dt)

    return cold_times, warm_times

def describe(name: str, times):
    print(f"{name}: mean={stats.mean(times)*1000:.2f} ms | p50={np.percentile(times,50)*1000:.2f} ms | p95={np.percentile(times,95)*1000:.2f} ms | n={len(times)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to any RGB image")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    assert Path(args.image).exists(), f"Image not found: {args.image}"
    print(f"Device: {args.device}")

    load_cold = time_clip_load(args.repeats, args.device)
    load_cached = time_cached_getter(args.repeats, args.device)
    describe("Load (clip.load each time)", load_cold)
    describe("Load (get_clip_model cached)", load_cached)

    cold, warm = time_embedding(args.image, args.repeats, args.device)
    describe("Embed (new model each time)", cold)
    describe("Embed (cached model reused)", warm)

    # Optional: GPU memory sanity check
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.empty_cache(); gc.collect()
        mem0 = torch.cuda.memory_allocated()

        models = []
        for _ in range(3):
            m, _ = clip.load("ViT-B/32", device=args.device)
            models.append(m)
        mem_after_three_fresh = torch.cuda.memory_allocated()
        del models; gc.collect(); torch.cuda.empty_cache()
        mem_after_cleanup = torch.cuda.memory_allocated()

        m1, _ = get_clip_model()
        mem_after_cached_one = torch.cuda.memory_allocated()
        m2, _ = get_clip_model()
        mem_after_cached_two = torch.cuda.memory_allocated()
        print(f"GPU mem (bytes): start={mem0}, after 3 fresh loads={mem_after_three_fresh}, "
              f"after cleanup={mem_after_cleanup}, cached first={mem_after_cached_one}, "
              f"cached second={mem_after_cached_two}")

if __name__ == "__main__":
    main()

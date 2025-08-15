import argparse
import os
import sys
import time
from pathlib import Path
import cProfile
import pstats
import io

try:
    import tracemalloc
except Exception:
    tracemalloc = None

try:
    import torch
    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
except Exception:
    torch = None

# make sure package imports work when run as script
if __package__ is None and __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from image_recommender.pipeline.search_pipeline import combined_similarity_search


def _ensure_profiles_dir() -> Path:
    p = Path("profiles")
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_cpu_profile(query_paths, index_path, mapping_path, k_clip=20, top_k=5, sort="cumulative"):
    out_dir = _ensure_profiles_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    prof_path = out_dir / f"cpu_{ts}.prof"
    txt_path  = out_dir / f"cpu_{ts}.txt"

    def target():
        return combined_similarity_search(
            query_paths,
            clip_index_path=index_path,
            clip_mapping_path=mapping_path,
            k_clip=k_clip,
            top_k_result=top_k
        )

    pr = cProfile.Profile()
    pr.enable()
    results = target()
    pr.disable()

    pr.dump_stats(str(prof_path))
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sort)
    ps.print_stats(40)  # top 40 lines
    with open(txt_path, "w") as f:
        f.write(s.getvalue())

    return results, prof_path, txt_path


def run_mem_profile(query_paths, index_path, mapping_path, k_clip=20, top_k=5, limit=30):
    if tracemalloc is None:
        raise RuntimeError("tracemalloc not available in this Python build")

    out_dir = _ensure_profiles_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    txt_path = out_dir / f"mem_{ts}.txt"

    tracemalloc.start()
    results = combined_similarity_search(
        query_paths,
        clip_index_path=index_path,
        clip_mapping_path=mapping_path,
        k_clip=k_clip,
        top_k_result=top_k
    )
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    with open(txt_path, "w") as f:
        f.write(f"Top {limit} memory consumers:\n")
        for stat in top_stats[:limit]:
            f.write(str(stat) + "\n")
        current, peak = tracemalloc.get_traced_memory()
        f.write(f"\nCurrent: {current/1e6:.2f} MB, Peak: {peak/1e6:.2f} MB\n")

    tracemalloc.stop()
    return results, txt_path


def run_torch_profile(query_paths, index_path, mapping_path, k_clip=20, top_k=5, trace_dir="profiles/torch"):
    if torch is None:
        raise RuntimeError("torch not available")

    out_dir = Path(trace_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    sch = schedule(wait=1, warmup=1, active=3, repeat=1)

    with profile(
        activities=activities,
        schedule=sch,
        on_trace_ready=tensorboard_trace_handler(str(out_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # one warmup + active steps as a single run (we just step the scheduler)
        combined_similarity_search(
            query_paths,
            clip_index_path=index_path,
            clip_mapping_path=mapping_path,
            k_clip=k_clip,
            top_k_result=top_k
        )
        for _ in range(6):
            prof.step()

    # Save a textual table as well
    ts = time.strftime("%Y%m%d_%H%M%S")
    txt_path = Path("profiles") / f"torch_{ts}.txt"
    with open(txt_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
    return txt_path, out_dir


def parse_args():
    ap = argparse.ArgumentParser(description="Profile combined_similarity_search.")
    ap.add_argument("--query", "-q", nargs="+", required=True, help="One or more query image paths")
    ap.add_argument("--index", "-i", required=True, help="clip_index.ann path")
    ap.add_argument("--mapping", "-m", required=True, help="index_to_id.json path")
    ap.add_argument("--k-clip", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--mode", choices=["cpu", "mem", "torch"], default="cpu")
    ap.add_argument("--sort", default="cumulative", help="cProfile sort (time, cumulative, tottime, ...)")
    return ap.parse_args()


def main():
    args = parse_args()
    query = args.query if len(args.query) > 1 else args.query[0]

    if args.mode == "cpu":
        results, prof_path, txt_path = run_cpu_profile(
            query, args.index, args.mapping, k_clip=args.k_clip, top_k=args.top_k, sort=args.sort
        )
        print(f"[CPU] .prof: {prof_path}")
        print(f"[CPU] Top stats: {txt_path}")
    elif args.mode == "mem":
        results, txt_path = run_mem_profile(
            query, args.index, args.mapping, k_clip=args.k_clip, top_k=args.top_k
        )
        print(f"[MEM] Stats: {txt_path}")
    else:
        txt_path, trace_dir = run_torch_profile(
            query, args.index, args.mapping, k_clip=args.k_clip, top_k=args.top_k
        )
        print(f"[TORCH] Table: {txt_path}")
        print(f"[TORCH] TensorBoard traces under: {trace_dir}")

    # minimal output of results
    if isinstance(query, list):
        print(f"Queries: {len(query)} images")
    else:
        print(f"Query: {query}")
    print("Top results:")
    for p, s in (results or [])[:args.top_k]:
        print(f"  {p}  {s:.4f}")


if __name__ == "__main__":
    main()

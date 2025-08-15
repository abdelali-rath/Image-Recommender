import argparse, cProfile, io, os, time
from pathlib import Path
import pstats
import textwrap
import matplotlib.pyplot as plt

import sys
if __package__ is None and __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from image_recommender.pipeline.search_pipeline import combined_similarity_search


def _plot_stats(stats: pstats.Stats, top: int = 25, title: str = "cProfile – Top functions",
                out: str | None = None):
    # stats.stats: { (filename, lineno, funcname): (cc, nc, tt, ct, callers) }
    items = []
    for (filename, lineno, funcname), (cc, nc, tt, ct, callers) in stats.stats.items():
        label = f"{Path(filename).name}:{lineno} {funcname}"
        items.append((label, tt, ct))

    # sort by cumulative time (ct) desc
    items.sort(key=lambda x: x[2], reverse=True)
    items = items[:top]

    if not items:
        print("No data to plot.")
        return

    labels, tt_list, ct_list = zip(*items)
    child_list = [max(0.0, ct - tt) for tt, ct in zip(tt_list, ct_list)]
    y = range(len(labels))

    plt.figure(figsize=(12, 0.45 * len(labels) + 2), constrained_layout=True)
    # Children time (ct - tt) first, then self time stacked on top
    plt.barh(y, child_list, label="children time (cum - self)")
    plt.barh(y, tt_list, left=child_list, label="self time")
    plt.gca().invert_yaxis()
    plt.yticks(y, [textwrap.shorten(l, width=70, placeholder="…") for l in labels])
    plt.xlabel("seconds")
    plt.title(title)
    plt.legend()

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        print(f"Saved plot → {out}")
    else:
        plt.show()


def _profile_run(query_paths, index_path, mapping_path, k_clip: int, top_k: int) -> pstats.Stats:
    pr = cProfile.Profile()
    pr.enable()
    _ = combined_similarity_search(
        query_paths,
        clip_index_path=index_path,
        clip_mapping_path=mapping_path,
        k_clip=k_clip,
        top_k_result=top_k,
    )
    pr.disable()
    return pstats.Stats(pr)


def parse_args():
    ap = argparse.ArgumentParser(description="Plot cProfile results with matplotlib.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prof", help="Path to an existing .prof file to visualize")
    mode.add_argument("--run", action="store_true", help="Profile a fresh run of the pipeline")

    ap.add_argument("--query", "-q", nargs="+", help="One or more query image paths (required for --run)")
    ap.add_argument("--index", "-i", help="clip_index.ann path (required for --run)")
    ap.add_argument("--mapping", "-m", help="index_to_id.json path (required for --run)")
    ap.add_argument("--k-clip", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--top", type=int, default=25, help="How many functions to show")
    ap.add_argument("--out", help="Output image path (PNG). If omitted, show window.")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.prof:
        stats = pstats.Stats(args.prof)
        title = f"cProfile – {Path(args.prof).name}"
    else:
        if not (args.query and args.index and args.mapping):
            raise SystemExit("--run requires --query, --index, and --mapping")
        query = args.query if len(args.query) > 1 else args.query[0]
        stats = _profile_run(query, args.index, args.mapping, args.k_clip, args.top_k)
        ts = time.strftime("%Y%m%d_%H%M%S")
        title = f"cProfile run {ts}"

    _plot_stats(stats, top=args.top, title=title, out=args.out)


if __name__ == "__main__":
    main()

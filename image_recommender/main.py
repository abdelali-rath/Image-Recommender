import argparse
from image_recommender.search_pipeline import combined_similarity_search
from image_recommender.visualize_results import show_image_results


def main():
    parser = argparse.ArgumentParser(description="Find similar images using combined CLIP, color, and pHash similarity.")
    parser.add_argument("input_image", type=str, help="Path to the input image")
    parser.add_argument("--index", type=str, default="clip_index.ann", help="Path to Annoy index file")
    parser.add_argument("--mapping", type=str, default="index_to_id.json", help="Path to index-to-ID mapping file")
    parser.add_argument("--topk", type=int, default=5, help="Number of results to return")
    parser.add_argument("--clipk", type=int, default=20, help="How many CLIP neighbors to consider")
    parser.add_argument("--visualize", action="store_true", help="Visualize results using matplotlib")

    args = parser.parse_args()

    results = combined_similarity_search(
        input_path=args.input_image,
        clip_index_path=args.index,
        clip_mapping_path=args.mapping,
        k_clip=args.clipk,
        top_k_result=args.topk
    )

    print(f"\n🔍 Top {args.topk} similar images for: {args.input_image}\n")
    for rank, (path, score) in enumerate(results, 1):
        print(f"{rank}. {score:.4f} → {path}")

    if args.visualize:
        show_image_results(args.input_image, results)


if __name__ == "__main__":
    main()

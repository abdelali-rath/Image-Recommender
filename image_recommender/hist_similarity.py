import argparse
from pathlib import Path
from loader import image_path_loader
from PIL import Image
import numpy as np



def load_and_resize(path, size=(64, 64)):

    img = Image.open(path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)
    # Returns a Numpy array of shape (height, width, 3) 3rd dimension are the RGB channels
    return np.asarray(img, dtype=np.float32) / 255.0


def extract_color_histogram(img_array, bins=(8, 8, 8)):

    pixels = img_array.reshape(-1, 3)
    hist, _ = np.histogramdd(pixels, bins=bins, range=[(0,1), (0,1), (0,1)])
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


def l2_distance(a, b):

    # Eucludian distance between two vectors
    return np.linalg.norm(a - b)


def compare_histograms(query_path, image_paths, size=(64,64), bins=(8,8,8)):
    """
    Compare a query image against a list of image_paths using
    euclidian distance on color histograms.
    """
    q_img = load_and_resize(query_path, size=size)
    q_hist = extract_color_histogram(q_img, bins=bins)

    results = []
    for path in image_paths:
        hist = extract_color_histogram(load_and_resize(path, size=size), bins=bins)
        dist = l2_distance(q_hist, hist)
        results.append((path, dist))

    return results


def top_k_hist(results, k=5):

    # Return top-k entries sorted by hist_dist ascending
    return sorted(results, key=lambda x: x[1])[:k]



def main():

    parser = argparse.ArgumentParser(description="Find top 5 similar images by color histogram")
    
    parser.add_argument(
        "--folder",
        type=str,
        default=r"C:\Users\meist\Downloads\random_pictures",        # Change path accordingly
        help="Path to the folder containing images"
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Path to the query image (if omitted, the first image in folder is used)"
    )

    parser.add_argument(
        "--bins", nargs=3, type=int, default=[8,8,8],
        help="Histogram bin counts for R, G, B channels"
    )

    parser.add_argument(
        "--size", nargs=2, type=int, default=[64,64],
        help="Resize dimensions (width height) for speed"
    )

    args = parser.parse_args()

    # Load image paths using loader
    paths = [path for (_id, path) in image_path_loader(args.folder)]
    if not paths:
        print(f"No images found in {args.folder}")
        return

    query = args.query or paths[0]
    if not Path(query).is_file():
        print(f"Query image {query} not found. Using first image in folder instead.")
        query = paths[0]

    print(f"Query image: {query}")
    results = compare_histograms(query, paths, size=tuple(args.size), bins=tuple(args.bins))
    top5 = top_k_hist(results, k=5)


    print("\nTop 5 by Color Histogram (L2) distance:\n")
    for path, dist in top5:
        print(f"{dist:.6f}\t{path}")



if __name__ == "__main__":
    main()


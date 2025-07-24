import matplotlib.pyplot as plt
from PIL import Image


def show_image_results(query_paths, result_paths_with_scores):
    """
    Displays input image(s) and top-k result images with similarity scores.
    query_paths: str or list of str
    result_paths_with_scores: list of (path, score)
    """
    if isinstance(query_paths, str):
        query_paths = [query_paths]

    num_queries = len(query_paths)
    top_k = len(result_paths_with_scores)

    fig, axes = plt.subplots(2, max(num_queries, top_k), figsize=(3 * max(num_queries, top_k), 6))

    # Plot query images
    for i, path in enumerate(query_paths):
        try:
            img = Image.open(path).convert("RGB")
            axes[0, i].imshow(img)
            axes[0, i].axis("off")
            axes[0, i].set_title(f"Query {i+1}")
        except Exception as e:
            axes[0, i].axis("off")
            axes[0, i].set_title(f"Query {i+1} (Error)")

    # Hide unused query axes if any
    for j in range(len(query_paths), max(num_queries, top_k)):
        axes[0, j].axis("off")

    # Plot result images
    for i, (path, score) in enumerate(result_paths_with_scores):
        try:
            img = Image.open(path).convert("RGB")
            axes[1, i].imshow(img)
            axes[1, i].axis("off")
            axes[1, i].set_title(f"{score:.4f}")
        except Exception as e:
            axes[1, i].axis("off")
            axes[1, i].set_title("Error")

    # Hide unused result axes if any
    for j in range(len(result_paths_with_scores), max(num_queries, top_k)):
        axes[1, j].axis("off")

    plt.tight_layout()
    plt.show()

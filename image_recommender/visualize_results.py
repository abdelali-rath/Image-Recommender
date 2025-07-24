import matplotlib.pyplot as plt
from PIL import Image


def show_image_results(query_path: str, results: list, max_width=224):
    """
    Displays the query image and its top results with scores.

    Args:
        query_path (str): Path to the input image
        results (list): List of tuples (image_path, score)
        max_width (int): Display width for each image
    """
    n = len(results) + 1  # query + results

    fig, axes = plt.subplots(1, n, figsize=(n * 3, 4))

    # Query image
    query_img = Image.open(query_path).convert("RGB")
    axes[0].imshow(query_img)
    axes[0].set_title("Query", fontsize=10)
    axes[0].axis("off")

    # Similar images
    for i, (path, score) in enumerate(results):
        try:
            img = Image.open(path).convert("RGB")
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f"{score:.3f}", fontsize=8)
            axes[i + 1].axis("off")
        except Exception as e:
            print(f"❌ Error loading image: {path} – {e}")
            axes[i + 1].axis("off")
            axes[i + 1].set_title("Error")

    plt.tight_layout()
    plt.show()

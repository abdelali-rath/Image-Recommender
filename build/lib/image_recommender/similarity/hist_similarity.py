import numpy as np
from PIL import Image


def compute_histogram(image: Image.Image, bins=8) -> np.ndarray:
    """
    Computes a normalized RGB histogram for the given PIL image.
    
    Args:
        image (PIL.Image): The input image (should be RGB)
        bins (int): Number of bins per channel

    Returns:
        np.ndarray: Normalized histogram vector of shape (3 * bins,)
    """
    image_array = np.asarray(image)
    histogram = []

    for channel in range(3):  # R, G, B
        hist, _ = np.histogram(
            image_array[:, :, channel],
            bins=bins,
            range=(0, 256),
            density=True  # normalize
        )
        histogram.extend(hist)

    return np.array(histogram, dtype=np.float32)


def image_color_similarity(img1: Image.Image, img2: Image.Image, bins=8) -> float:
    """
    Compares two images based on their color histograms using L2 distance.
    
    Args:
        img1, img2 (PIL.Image): RGB images to compare
        bins (int): Number of bins per channel

    Returns:
        float: The L2 distance between histograms (lower = more similar)
    """
    hist1 = compute_histogram(img1, bins)
    hist2 = compute_histogram(img2, bins)

    # Euclidean (L2) distance
    distance = np.linalg.norm(hist1 - hist2)
    return distance

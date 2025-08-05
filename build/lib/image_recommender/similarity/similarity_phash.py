from PIL import Image
import imagehash


def compute_phash(image: Image.Image) -> imagehash.ImageHash:
    """
    Computes the perceptual hash (pHash) of an image.

    Args:
        image (PIL.Image): RGB input image

    Returns:
        imagehash.ImageHash: Hash object (64-bit)
    """
    return imagehash.phash(image)


def phash_similarity(img1: Image.Image, img2: Image.Image) -> int:
    """
    Compares two images using perceptual hash (pHash) and returns Hamming distance.

    Args:
        img1, img2 (PIL.Image): RGB images to compare

    Returns:
        int: Hamming distance (0 = identical, higher = less similar)
    """
    hash1 = compute_phash(img1)
    hash2 = compute_phash(img2)

    return hash1 - hash2  # built-in Hamming distance

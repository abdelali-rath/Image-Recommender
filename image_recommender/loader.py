import os
import hashlib
from PIL import Image

from image_recommender.database import create_table, insert_image_data


def load_image(path):
    """
    Loads an image from disk and returns a PIL Image object in RGB format.
    """
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as e:
        print(f"❌ Error loading image {path}: {e}")
        return None


def preprocess_image(image, size=(224, 224)):
    """
    Resize and normalize the image.
    Returns a resized PIL image.
    """
    return image.resize(size)


def load_images_generator(dataset_path, extensions={".jpg", ".jpeg", ".png"}):
    """
    Generator that yields valid image file paths from a directory (including subfolders).
    """
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                full_path = os.path.join(root, file)
                yield full_path


def generate_image_id(path):
    """
    Generates a unique ID based on the image path using SHA256.
    """
    return hashlib.sha256(path.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    dataset_path = "/Volumes/BigData06/data"

    # Ensure the database table exists
    create_table()

    count = 0
    for path in load_images_generator(dataset_path):
        img = load_image(path)
        if img:
            resized = preprocess_image(img)
            image_id = generate_image_id(path)
            width, height = resized.size

            # Insert into the database
            insert_image_data(image_id, path, width, height)

            print(f"[{count}] ✅ Stored {path} → ID: {image_id}")
            count += 1

        if count >= 5:  # Test only the first 5 images
            break

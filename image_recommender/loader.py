import os
from PIL import Image
import itertools
from pathlib import Path


def image_path_loader(root_dir, extensions=('jpg', 'jpeg', 'png', 'bmp', 'tiff')):

    id_counter = itertools.count(1)
    root_path = Path(root_dir)

    # rglob returns all matching file paths
    for file_path in root_path.rglob('*'):
        if file_path.is_file() \
        and file_path.suffix.lower().lstrip('.') in extensions:
            yield next(id_counter), str(file_path)


if __name__ == "__main__":
    folder = r"C:\Users\meist\Downloads\random_pictures"        # Change path accordingly
    for img_id, img_path in image_path_loader(folder):
        print(f"{img_id:4d}: {img_path}")
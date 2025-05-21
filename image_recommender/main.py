'''
from loader import image_path_loader
from PIL import Image

for img_id, path in image_path_loader(r"C:\Users\meist\Downloads\random_pictures"):
    # only open if this image is a candidate
    img = Image.open(path)
    # … process histogram, embeddings, etc.
'''
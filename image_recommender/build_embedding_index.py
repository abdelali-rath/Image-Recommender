# embed_images.py

import os
import sys
import json
from tqdm import tqdm
from PIL import Image

# Add project root to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_recommender.database import connect_db

def get_all_images_from_db():
    """
    Returns a list of (image_id, path) from the SQLite database.
    """
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, path FROM images;")
        return cursor.fetchall()

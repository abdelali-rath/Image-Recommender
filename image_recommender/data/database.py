import sqlite3
from typing import Optional, Tuple

# Path to the local SQLite database file
DB_PATH = "image_metadata.db"  # You can externalize this into config if needed


def connect_db():
    """
    Opens a connection to the SQLite database.
    """
    return sqlite3.connect(DB_PATH)


def create_table():
    """
    Creates the 'images' table in the database if it doesn't already exist.
    Columns:
        - id: unique image ID (primary key)
        - path: file path to the image
        - width, height: image dimensions in pixels
    """
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                width INTEGER,
                height INTEGER
            );
        """)
        conn.commit()


def insert_image_data(image_id: str, path: str, width: int, height: int):
    """
    Inserts metadata for a single image into the database.
    Duplicate entries (by ID) are ignored.

    Args:
        image_id (str): Unique SHA256 ID of the image
        path (str): Full file path to the image
        width (int): Image width in pixels
        height (int): Image height in pixels
    """
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO images (id, path, width, height)
            VALUES (?, ?, ?, ?);
        """, (image_id, path, width, height))
        conn.commit()


def get_image_by_id(image_id: str) -> Optional[Tuple[str, int, int]]:
    """
    Retrieves image metadata by ID.

    Args:
        image_id (str): The unique ID of the image to look up

    Returns:
        Tuple of (path, width, height) if found, or None
    """
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT path, width, height FROM images WHERE id = ?;
        """, (image_id,))
        return cursor.fetchone()

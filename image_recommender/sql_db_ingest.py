'''

import sqlite3
from pathlib import Path
from PIL import Image



# Constants
DB_PATH = "images_meta.db"
ROOT_DIR = r"D:\data"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
BATCH_SIZE = 1000



def init_db(db_path: str) -> sqlite3.Connection:

    # Connect to SQL and create images table 
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON images(file_path);")
    conn.commit()
    return conn


def ingest_images(root_dir: str, conn: sqlite3.Connection):

    # Read image metadata and insert into database (resolution/size)
    batch = []
    for path in Path(root_dir).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            # Skip corrupted or unsupported images
            continue

        batch.append((str(path), width, height))

        # When batch is large, write to DB
        if len(batch) >= BATCH_SIZE:
            conn.executemany(
                "INSERT OR IGNORE INTO images (file_path, width, height) VALUES (?, ?, ?);",
                batch
            )
            conn.commit()
            batch.clear()

    # Insert remaining records
    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO images (file_path, width, height) VALUES (?, ?, ?);",
            batch
        )
        conn.commit()


def get_all_image_paths(conn: sqlite3.Connection):

    # Image paths
    cursor = conn.cursor()
    for (file_path,) in cursor.execute("SELECT file_path FROM images;"):
        yield file_path


if __name__ == "__main__":
    
    # 1. Initialize DB (creates file in working directory)
    conn = init_db(DB_PATH)

    # 2. Ingest images from disk into the database
    print(f"Ingesting images from {ROOT_DIR} into {DB_PATH} ...")
    ingest_images(ROOT_DIR, conn)
    print("Ingestion complete.")

    # 3. Example query: count total images
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM images;")
    total = cursor.fetchone()[0]
    print(f"Total images stored in DB: {total}")

    conn.close()

'''
# Resize images to 64 x 64, therefore creating "thumbnails" to make the db faster

import sqlite3
from pathlib import Path
from PIL import Image
import sys



# Configuration
DB_PATH     = "images_meta.db"
SRC_ROOT    = Path(r"D:\data")
THUMB_ROOT  = Path(r"D:\data_thumbs")
THUMB_SIZE  = (64, 64)
EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
BATCH_SIZE  = 1000
BAR_LENGTH  = 40   # characters for the ASCII bar


def init_db(db_path: str) -> sqlite3.Connection:

    
    # Create (or open) the images table. "orig_path" is UNIQUE so we can skip already-processed images on re-run.
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS images ("
                "  id         INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  orig_path  TEXT    UNIQUE NOT NULL,"
                "  thumb_path TEXT    UNIQUE NOT NULL,"
                "  width      INTEGER NOT NULL,"
                "  height     INTEGER NOT NULL"
                ");")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orig ON images(orig_path);")
    conn.commit()
    return conn


def print_progress(current, total):

    """
    Prints an ASCII progress bar to stdout.
    """
    percent = current / total if total else 1
    filled = int(BAR_LENGTH * percent)
    bar = "#" * filled + "-" * (BAR_LENGTH - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total} ({percent:.1%})")
    sys.stdout.flush()


def ingest_thumbnails(src_root: Path, thumb_root: Path, conn: sqlite3.Connection):

    # 1. Preload already-processed paths
    cur = conn.cursor()
    cur.execute("SELECT orig_path FROM images;")
    done = set(row[0] for row in cur.fetchall())

    # 2. Gather only the unprocessed image paths
    all_paths = [p for p in src_root.rglob("*")
                 if p.is_file() and p.suffix.lower() in EXTENSIONS]
    to_process = [p for p in all_paths if str(p) not in done]
    total = len(to_process)

    print(f"Found {len(all_paths)} images, {len(done)} already done, {total} to process.")
    if total == 0:
        print("Nothing to do.")
        return

    thumb_root.mkdir(parents=True, exist_ok=True)
    batch = []
    processed = 0

    try:
        for src_path in to_process:
            processed += 1
            print_progress(processed, total)

            # Compute thumbnail path
            rel = src_path.relative_to(src_root)
            thumb_path = thumb_root / rel.with_suffix(".jpg")
            thumb_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    thumb = img.resize(THUMB_SIZE, Image.BILINEAR)
                    thumb.save(thumb_path, format="JPEG", quality=85)
                    w, h = thumb.size
            except Exception as e:
                # log skip but continue
                sys.stdout.write(f" ▶ Skipped: {src_path.name}")
                sys.stdout.flush()
                continue

            batch.append((str(src_path), str(thumb_path), w, h))

            # Batch insert
            if len(batch) >= BATCH_SIZE:
                conn.executemany(
                    "INSERT OR IGNORE INTO images "
                    "(orig_path, thumb_path, width, height) VALUES (?, ?, ?, ?);",
                    batch
                )
                conn.commit()
                batch.clear()

        # Final batch
        if batch:
            conn.executemany(
                "INSERT OR IGNORE INTO images "
                "(orig_path, thumb_path, width, height) VALUES (?, ?, ?, ?);",
                batch
            )
            conn.commit()

    except KeyboardInterrupt:
        # On Ctrl+C commit what’s left and exit gracefully
        if batch:
            conn.executemany(
                "INSERT OR IGNORE INTO images "
                "(orig_path, thumb_path, width, height) VALUES (?, ?, ?, ?);",
                batch
            )
            conn.commit()
        print("\nInterrupted by user. Progress saved.")
        return

    # Finish up
    sys.stdout.write("\n")
    print("Thumbnail ingestion complete.")


if __name__ == "__main__":

    conn = init_db(DB_PATH)
    print(f"Starting thumbnail ingestion from {SRC_ROOT} → {THUMB_ROOT}")
    ingest_thumbnails(SRC_ROOT, THUMB_ROOT, conn)
    total_done = conn.execute("SELECT COUNT(*) FROM images;").fetchone()[0]
    print(f"Total images recorded in DB: {total_done}")
    conn.close()

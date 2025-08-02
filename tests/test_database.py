import sqlite3
import os
import pytest
from image_recommender.data import database


def test_create_and_query_table(tmp_path, monkeypatch):
    # Redirect DB_PATH to a temporary SQLite file
    db_file = tmp_path / "test_image_metadata.db"
    monkeypatch.setattr(database, 'DB_PATH', str(db_file))

    # Ensure no DB exists initially
    assert not db_file.exists()

    # Create table
    database.create_table()
    assert db_file.exists()

    # Verify table 'images' exists
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images';")
    assert cursor.fetchone() is not None
    conn.close()


def test_insert_and_get_image(tmp_path, monkeypatch):
    # Setup a fresh DB
    db_file = tmp_path / "test_image_metadata.db"
    monkeypatch.setattr(database, 'DB_PATH', str(db_file))
    database.create_table()

    # Insert and retrieve image metadata
    image_id = 'img123'
    path = '/path/to/image.jpg'
    width, height = 100, 200
    database.insert_image_data(image_id, path, width, height)
    result = database.get_image_by_id(image_id)
    assert result == (path, width, height)

    # Querying non-existent ID should return None
    assert database.get_image_by_id('does_not_exist') is None

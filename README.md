# Project "Image Recommender"

<img width="500" height="500" alt="Image Recommender (4)-modified" src="https://github.com/user-attachments/assets/81988c4d-d2c7-4b5d-9163-5a6e4de60e9e" />

## DAISY course 4th semester "Big Data Engineering"

**Find the top-5 visually similar images using multiple metrics and fast vector indexing.**

---<img width="500" height="500" alt="Image Recommender (4)-modified" src="https://github.com/user-attachments/assets/81988c4d-d2c7-4b5d-9163-5a6e4de60e9e" />


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Testing](#testing)
- [License](#license)

---

## Overview

This project implements a scalable and modular image similarity search system designed for large-scale datasets (~500,000+ images).  
It supports multiple similarity metrics and fast search via the Annoy index.

---

## Features

- CLIP embedding similarity (semantic)
- Color histogram-based similarity (L2 distance)
- Perceptual Hashing (pHash)
- Weighted combination of all metrics
- Annoy index for fast approximate nearest neighbor search
- Visualization of query and result images
- Extensible modular codebase
- Works on large datasets via streaming + SQLite
- Clean CLI interface

---

## Project Structure

```
Image-Recommender/
├── config/
│   └── default_config.yaml                     # Optional configuration
│
├── image_recommender/                          # Python package
│   ├── data/
│   │   ├── db/
│   │   │   └── image_metadata.db               # SQLite DB with image paths & metadata
│   │   ├── out/
│   │   │   ├── clip_index.ann                  # Annoy index for CLIP
│   │   │   └── index_to_id.json                # Mapping: Annoy index → DB ID
│   │   ├── database.py                         # DB query + connect logic
│   │   └── loader.py                           # Image loading & preprocessing
│   │
│   ├── pipeline/
│   │   ├── build_embedding_index.py            # Compute embeddings & build Annoy index
│   │   ├── query_clip_similar.py               # CLIP-only query tool
│   │   ├── search_pipeline.py                  # Combined similarity logic
│   │   └── visualize_results.py                # Plotting of query results
│   │
│   ├── similarity/
│   │   ├── hist_similarity.py                  # Color histogram similarity (L2)
│   │   ├── similarity_embedding.py             # CLIP logic + Annoy I/O
│   │   └── similarity_phash.py                 # Perceptual hash similarity
│   │
│   └── __init__.py
│
├── tests/
│   ├── test_database.py                        # Unit tests: DB
│   ├── test_loader.py                          # Unit tests: image loader
│   └── test_similarity.py                      # Unit tests: similarity measures
│
├── main.py                                     # Entry point CLI (calls pipeline)
├── requirements.txt                            # All dependencies
├── README.md
├── LICENSE
└── .github/workflows/python-package-conda.yml  # GitHub Actions CI


```

---

## Installation

```bash
git clone https://github.com/abdelali-rath/Image-recommender.git
cd Image-recommender
pip install -r requirements.txt
```

---

## Setup

Before running any image similarity search, you must **initialize two components**:


#### 1. Load image metadata into the database

Use `loader.py` to scan a directory and insert image metadata (ID, path, dimensions) into the SQLite database.

Run:

```bash
python -m image_recommender.data.loader
```

This will:

* Traverse all image files under a specified folder (default path is hardcoded)
* Store each image’s path, width, and height in `data/db/image_metadata.db`

#### 2. Build CLIP embedding index (Annoy + Mapping)

Next, use `build_embedding_index.py` to compute CLIP embeddings and build an Annoy index.

Run:

```bash
python -m image_recommender.pipeline.build_embedding_index
```

This will:

* Load image paths from the database
* Compute CLIP embeddings for each image
* Build an Annoy index for fast nearest-neighbor search
* Save:

  * `clip_index.ann`: the Annoy index file
  * `index_to_id.json`: a mapping from Annoy index position to image ID

Output is written to: `image_recommender/data/out/`

> **Warning:** Embedding 500k+ images can take **many hours** depending on your hardware. You can limit the number of processed images by setting `max_images = 500` or similar in the script.

Once both steps are complete, your system is ready to run efficient multimodal similarity queries.

---


## Usage

```bash
# Run with one or more input images
python -m image_recommender.main \
path/to/image1.jpg \
path/to/image2.jpg \
  --topk 5 \
  --clipk 20 \
  --visualize
```

Optional CLI flags:

* `--topk`: number of final results to show (default: 5)
* `--clipk`: number of CLIP neighbors to consider (default: 20)
* `--visualize`: show input + result images via matplotlib

---

## Testing

```bash
pytest
```
---

## License

Distributed under the MIT License. See `LICENSE` file for details.

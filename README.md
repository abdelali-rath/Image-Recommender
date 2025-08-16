# Image Recommender

<img width="100" height="100" alt="Image Recommender Logo" src="image_recommender/assets/logo.png" />

## DAISY course 4th semester "Big Data Engineering"

**Find the top-5 visually similar images using multiple metrics and fast vector indexing.**

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [GUI Application](#gui-application)
- [Tools](#tools)
- [Testing](#testing)
- [Development](#development)
- [License](#license)

---

## Overview

This project implements a scalable and modular image similarity search system designed for large-scale datasets (~500,000+ images).
It supports multiple similarity metrics and fast search via the Annoy index, with both command-line and graphical user interfaces.

---

## Features

- **Multiple Similarity Metrics:**

  - CLIP embedding similarity (semantic)
  - Color histogram-based similarity (L2 distance)
  - Perceptual Hashing (pHash)
  - Weighted combination of all metrics
- **Performance & Scalability:**

  - Annoy index for fast approximate nearest neighbor search
  - Works on large datasets via streaming + SQLite
  - Caching and optimization tools
- **User Interfaces:**

  - Clean CLI interface
  - Modern PyQt5 GUI application
  - Visualization of query and result images
- **Development Features:**

  - Extensible modular codebase
  - Comprehensive testing suite
  - Benchmarking and profiling tools
  - Modern Python packaging with `pyproject.toml`

---

## Project Structure

```
Image-Recommender/
├── config/
│   └── default_config.txt                   # Configuration file
│
├── image_recommender/                       # Python package
│   ├── assets/                              # GUI assets
│   │   ├── app_background.jpg               # Background image
│   │   └── logo.png                         # Application logo
│   │
│   ├── data/
│   │   ├── db/
│   │   │   └── image_metadata.db            # SQLite DB with image paths & metadata
│   │   ├── out/
│   │   │   ├── clip_index.ann               # Annoy index for CLIP
│   │   │   └── index_to_id.json             # Mapping: Annoy index → DB ID
│   │   ├── database.py                      # DB query + connect logic
│   │   └── loader.py                        # Image loading & preprocessing
│   │
│   ├── pipeline/
│   │   ├── build_embedding_index.py         # Compute embeddings & build Annoy index
│   │   ├── query_clip_similar.py            # CLIP-only query tool
│   │   ├── search_pipeline.py               # Combined similarity logic
│   │   └── visualize_results.py             # Plotting of query results
│   │
│   ├── similarity/
│   │   ├── hist_similarity.py               # Color histogram similarity (L2)
│   │   ├── similarity_embedding.py          # CLIP logic + Annoy I/O
│   │   └── similarity_phash.py              # Perceptual hash similarity
│   │
│   ├── tools/                               # Development and benchmarking tools
│   │   ├── bench_clip_batch.py              # CLIP batch processing benchmarks
│   │   ├── bench_clip_cache.py              # CLIP caching performance tests
│   │   ├── profiler.py                      # Performance profiling utilities
│   │   └── profile_plot.py                  # Performance visualization
│   │
│   ├── app.py                               # PyQt5 GUI application
│   ├── main.py                              # CLI entry point
│   └── __init__.py
│
├── tests/
│   ├── test_clip_batch.py                   # CLIP batch processing tests
│   ├── test_clip_model_cache.py             # CLIP model caching tests
│   ├── test_database.py                     # Unit tests: DB
│   ├── test_loader.py                       # Unit tests: image loader
│   └── test_similarity.py                   # Unit tests: similarity measures
│
├── pyproject.toml                           # Modern Python packaging configuration
├── requirements.txt                         # Dependencies
├── README.md
└── LICENSE
```

---

## Installation

### Option 1: Install from source

```bash
git clone https://github.com/abdelali-rath/Image-recommender.git
cd Image-recommender
pip install -e .
```

### Option 2: Install with GUI support

```bash
git clone https://github.com/abdelali-rath/Image-recommender.git
cd Image-recommender
pip install -e ".[qt]"
```

### Option 3: Install with development dependencies

```bash
git clone https://github.com/abdelali-rath/Image-recommender.git
cd Image-recommender
pip install -e ".[dev]"
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
* Store each image's path, width, and height in `data/db/image_metadata.db`

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

### Command Line Interface

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
* `--index`: path to Annoy index file (default: `image_recommender/data/out/clip_index.ann`)
* `--mapping`: path to index-to-ID mapping file (default: `image_recommender/data/out/index_to_id.json`)

### Using the installed command

If you installed the package, you can also use:

```bash
image-recommender path/to/image.jpg --topk 10 --visualize
```

---

## GUI Application

The project includes a modern PyQt5-based graphical user interface for easy image similarity search.

### Launching the GUI

```bash
# If installed with GUI support
python -m image_recommender.app

# Or using the module directly
python image_recommender/app.py
```

### GUI Features

- **Drag & Drop**: Simply drag images into the application window
- **File Browser**: Use the file browser to select input images
- **Real-time Search**: See results update as you add images
- **Visual Results**: View both input and result images side by side
- **Score Display**: See similarity scores for each result
- **Modern Interface**: Clean, intuitive design with custom styling

---

## Tools

The project includes several development and benchmarking tools:

### Performance Profiling

```bash
# Run performance profiling
python -m image_recommender.tools.profiler

# Generate performance plots
python -m image_recommender.tools.profile_plot
```

### CLIP Benchmarking

```bash
# Test CLIP batch processing performance
python -m image_recommender.tools.bench_clip_batch

# Test CLIP caching performance
python -m image_recommender.tools.bench_clip_cache
```

---

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_similarity.py
pytest tests/test_database.py
pytest tests/test_loader.py
pytest tests/test_clip_batch.py
pytest tests/test_clip_model_cache.py
```

---

## Development

### Code Quality

The project uses modern Python development tools:

- **Ruff**: Fast Python linter and formatter
- **Pytest**: Testing framework
- **Modern Packaging**: `pyproject.toml` configuration

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run linting and formatting
ruff check .
ruff format .

# Run tests with coverage
pytest --cov=image_recommender
```

### Building the Package

```bash
# Build the package
python -m build

# Install from built wheel
pip install dist/image_recommender-*.whl
```

---

## License

Distributed under the MIT License. See `LICENSE` file for details.

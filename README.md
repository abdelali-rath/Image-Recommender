# Project "Image Recommender"
## DAISY course 4th semester "Big Data Engineering"


**Find the top-5 visually similar images using multiple metrics and fast indexing.**

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project implements an image similarity search system for over ~500k images.

## Features
- Color-based histogram similarity
- Unit tests for all core components

## Repository Structure

```
image_recommender/
├── config/
│   └── default_config.yaml
├── image_recommender/
│   ├── __init__.py
│   ├── main.py
│   ├── loader.py                       # Image loading generator
│   ├── hist_similarity.py              # Color histogram-based similarity
│   ├── thumbnail_ingest.py              # Resizing to 64 x 64 and SQL database
├── tests/
│   ├── test.py
├── LICENSE
├── README.md
```

## Installation
```bash
git clone https://github.com/abdelali-rath/image-recommender.git
cd image-recommender
python -m pip install
# or on mac: python3 -m pip install
```

## Usage
```
python main.py
```

## Testing

```
pytest
```

## Contributing

1. Fork the repo

2. Create a feature branch

3. Submit a pull request

## License

Distributed under the MIT License. See LICENSE for details.

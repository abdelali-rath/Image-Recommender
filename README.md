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
├── image_recommender/
│   ├── __init__.py
│   ├── main.py
│   ├── loader.py                       # Image loading generator
│   ├── similarity_color.py             # Color histogram-based similarity
│   ├── search_pipeline.py              # End-to-end search logic
│   ├── database.py                     # SQL
├── config/
│   └── default_config.yaml
├── tests/
│   ├── test.py
├── profiling/
│   ├── profiler.py
├── environment.yml
├── setup.py
├── README.md
├── LICENSE
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

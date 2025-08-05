from setuptools import setup, find_packages

setup(
    name="image_recommender",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "Pillow>=11.3.0",
        "tqdm>=4.65.0",
        "ImageHash>=4.3.2",
        "annoy>=1.17.3",
        "torch>=2.3.0",
        # PEP 508 “direct URL” for CLIP
        "clip @ git+https://github.com/openai/CLIP.git@main#egg=clip",
    ],
)

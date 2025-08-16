import os
import torch
import clip
from PIL import Image
from annoy import AnnoyIndex
from typing import List


# Set device: use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
EMBEDDING_DIM = model.visual.output_dim


def compute_clip_embedding(image: Image.Image) -> torch.Tensor:
    """
    Computes the CLIP embedding for a given PIL image.

    Args:
        image (PIL.Image): RGB image

    Returns:
        torch.Tensor: Embedding vector (e.g., shape (512,))
    """
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze().cpu()


def build_annoy_index(embeddings: dict, index_path: str, n_trees: int = 10):
    """
    Builds and saves an Annoy index from given embeddings.

    Args:
        embeddings (dict): {image_id: np.array or list}
        index_path (str): Path to save the Annoy index
        n_trees (int): Number of trees (higher = better accuracy, slower build)
    """
    index = AnnoyIndex(EMBEDDING_DIM, metric="angular")
    for i, (image_id, vector) in enumerate(embeddings.items()):
        index.add_item(i, vector)
    index.build(n_trees)
    index.save(index_path)


def load_annoy_index(index_path: str) -> AnnoyIndex:
    """
    Loads an Annoy index from file.

    Returns:
        AnnoyIndex: Loaded index
    """
    index = AnnoyIndex(EMBEDDING_DIM, metric="angular")
    index.load(index_path)
    return index


def query_similar(image: Image.Image, index: AnnoyIndex, top_k=5) -> list:
    """
    Finds the top_k most similar items to the input image using CLIP + Annoy.

    Returns:
        List of (index, distance)
    """
    embedding = compute_clip_embedding(image)
    return index.get_nns_by_vector(embedding.tolist(), top_k, include_distances=True)

# Cache for CLIP model
_model_cache = {}
_preprocess_cache = {}

def get_clip_model():
    """Singleton pattern for the CLIP model"""
    if 'model' not in _model_cache:
        _model_cache['model'], _model_cache['preprocess'] = clip.load("ViT-B/32", device=device)
        _model_cache['model'].eval()
    return _model_cache['model'], _model_cache['preprocess']

def compute_clip_embeddings_batch(images: List[Image.Image]) -> torch.Tensor:
    """
    Compute normalized CLIP embeddings for a list of PIL images.
    Returns a (N, EMBEDDING_DIM) float32 tensor on CPU.
    """
    if not images:
        return torch.empty(0, EMBEDDING_DIM, dtype=torch.float32)

    model, preprocess = get_clip_model()
    dev = next(model.parameters()).device
    batch_inputs = torch.stack([preprocess(img) for img in images]).to(dev)

    with torch.inference_mode():
        embeddings = model.encode_image(batch_inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.detach().cpu().to(torch.float32)
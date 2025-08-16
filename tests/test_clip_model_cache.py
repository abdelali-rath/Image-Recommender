import numpy as np
import torch
from PIL import Image
import clip

from image_recommender.similarity.similarity_embedding import get_clip_model


def test_same_object_identity():
    m1, p1 = get_clip_model()
    m2, p2 = get_clip_model()
    assert m1 is m2
    assert p1 is p2


def _embed(model, preprocess, image):
    img = preprocess(image).unsqueeze(0).to(next(model.parameters()).device)
    with torch.inference_mode():
        emb = model.encode_image(img)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy()


def test_embeddings_equal(tmp_path):
    # Make a small dummy image
    img = Image.new("RGB", (64, 64), (123, 45, 67))

    # Fresh load
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fresh_model, fresh_pre = clip.load("ViT-B/32", device=dev)
    fresh_model.eval()
    fresh_emb = _embed(fresh_model, fresh_pre, img)

    # Cached
    cached_model, cached_pre = get_clip_model()
    cached_emb = _embed(cached_model, cached_pre, img)

    # Same numerical result within tolerance
    assert np.allclose(fresh_emb, cached_emb, atol=1e-6)

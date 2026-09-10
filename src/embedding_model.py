"""Shared SentenceTransformer loader so the retriever and the embedder agree on
device, backend and offline behaviour instead of each rolling their own.
"""

import os

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# torch | onnx | openvino. onnx/openvino are markedly faster on CPU but need
#   pip install "sentence-transformers[onnx]"      (or [openvino])
# Falls back to torch if the backend's runtime is not installed.
_BACKEND = os.getenv("EMBED_BACKEND", "torch").strip().lower()


def pick_device() -> str:
    return _DEVICE


def load_embedding_model(model_name: str, local_files_only: bool = True) -> SentenceTransformer:
    """Load `model_name` on the available device, honouring EMBED_BACKEND.

    `local_files_only=True` skips a Hugging Face network check at startup, which
    also lets the app run offline once the model is cached. onnx/openvino
    backends may need a one-time export, so they allow a download.
    """
    backend = _BACKEND if _BACKEND in {"torch", "onnx", "openvino"} else "torch"
    offline = local_files_only and backend == "torch"

    try:
        model = SentenceTransformer(
            model_name,
            device=_DEVICE,
            backend=backend,
            local_files_only=offline,
        )
        logger.info(f"Loaded embedding model '{model_name}' | backend={backend} | device={_DEVICE}")
        return model
    except Exception as error:
        if backend == "torch":
            raise
        logger.warning(
            f"EMBED_BACKEND={backend} unavailable ({error}); falling back to torch. "
            f'Install it with: pip install "sentence-transformers[{backend}]"'
        )
        model = SentenceTransformer(model_name, device=_DEVICE, local_files_only=local_files_only)
        logger.info(f"Loaded embedding model '{model_name}' | backend=torch | device={_DEVICE}")
        return model

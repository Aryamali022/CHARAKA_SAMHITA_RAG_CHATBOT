"""The embedding model: turns text into vectors for meaning-based search.

Model: BAAI/bge-small-en-v1.5 (384 dimensions, 512-token input), run locally
through fastembed (ONNX, no PyTorch). It is free, needs no API key, keeps the
data on this machine, and gives the same vectors on every run. The model is
downloaded once into storage/models/.

This is the only module that talks to fastembed, so tests can use a fake.
"""
from __future__ import annotations

import numpy as np

from src import config

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE models search better when a short question carries this instruction;
# passages are embedded without it.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, cache_dir=config.MODELS_DIR):
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer

        self.model_name = model_name
        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir))
        # A copy of the model's tokenizer without truncation, so long texts are
        # counted in full instead of being cut silently at 512 tokens.
        self._tokenizer = Tokenizer.from_str(self._model.model.tokenizer.to_str())
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def count_tokens(self, text: str) -> int:
        """Tokens of text, without the [CLS]/[SEP] the model adds around it."""
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = np.array(list(self._model.embed(texts, batch_size=batch_size)), dtype=np.float32)
        return _normalise(vectors)

    def embed_query(self, question: str) -> np.ndarray:
        [vector] = self._model.embed([QUERY_INSTRUCTION + question])
        return _normalise(np.asarray(vector, dtype=np.float32)[None, :])[0]


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Unit length, so a dot product is the cosine similarity."""
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

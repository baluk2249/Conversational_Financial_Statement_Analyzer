"""FAISS-backed narrative retrieval with metadata filtering before ranking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, Sequence

import faiss
import numpy as np
from numpy.typing import NDArray

from models.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class NarrativeStoreError(RuntimeError):
    """Raised when narrative evidence cannot be indexed, persisted, or searched."""


class TextEmbedder(Protocol):
    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class Model2VecEmbedder:
    """Lazy adapter for static embeddings that do not require PyTorch or ONNX."""

    def __init__(self, model_name: str = "minishlab/potion-base-8M") -> None:
        try:
            from model2vec import StaticModel

            self._model = StaticModel.from_pretrained(model_name)
        except Exception as exc:
            raise NarrativeStoreError(
                "Install the 'semantic' extra and check model access to use the default Model2Vec embedder"
            ) from exc

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors = self._model.encode(list(texts))
        return np.asarray(vectors, dtype=np.float32)


class FaissNarrativeStore:
    """Persistent narrative vector store; exact financial values remain in SQLite."""

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        embedder: TextEmbedder | None = None,
        model_name: str = "minishlab/potion-base-8M",
    ) -> None:
        self.embedder = embedder or Model2VecEmbedder(model_name)
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self._chunks: dict[str, DocumentChunk] = {}
        self._vectors: dict[str, NDArray[np.float32]] = {}
        self._index_cache: dict[tuple[str, str | None, str | None], tuple[faiss.Index, list[str]]] = {}
        if self.persist_directory is not None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._load()

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        try:
            vectors = self._normalized(self.embedder.encode([chunk.text for chunk in chunks]))
            if len(vectors) != len(chunks):
                raise ValueError("Embedder returned a different number of vectors than input chunks")
            if self._vectors and vectors.shape[1] != next(iter(self._vectors.values())).shape[0]:
                raise ValueError("Embedding dimension changed; use one model for the lifetime of an index")
            for chunk, vector in zip(chunks, vectors, strict=True):
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = vector.copy()
            self._index_cache.clear()
            self._persist()
        except Exception as exc:
            if isinstance(exc, NarrativeStoreError):
                raise
            raise NarrativeStoreError("Could not index narrative chunks in FAISS") from exc

    def search(
        self,
        query: str,
        company_id: str,
        period: str | None = None,
        document_type: str | None = None,
        top_k: int = 5,
    ) -> list[DocumentChunk]:
        if top_k <= 0:
            return []
        try:
            cache_key = (company_id, period, document_type)
            cached = self._index_cache.get(cache_key)
            if cached is None:
                candidate_ids = [
                    chunk_id for chunk_id, chunk in self._chunks.items()
                    if chunk.company_id == company_id
                    and (period is None or chunk.period == period)
                    and (document_type is None or chunk.document_type == document_type)
                ]
                if not candidate_ids:
                    return []
                dimension = int(self._vectors[candidate_ids[0]].shape[0])
                scoped_index = faiss.IndexFlatIP(dimension)
                scoped_index.add(np.stack([self._vectors[chunk_id] for chunk_id in candidate_ids]))
                cached = (scoped_index, candidate_ids)
                self._index_cache[cache_key] = cached

            query_vector = self._normalized(self.embedder.encode([query]))
            index, candidate_ids = cached
            scores, positions = index.search(query_vector, min(top_k, len(candidate_ids)))
            del scores
            return [
                self._chunks[candidate_ids[int(position)]]
                for position in positions[0]
                if 0 <= position < len(candidate_ids)
            ]
        except Exception as exc:
            if isinstance(exc, NarrativeStoreError):
                raise
            raise NarrativeStoreError("FAISS narrative search failed") from exc

    @staticmethod
    def _normalized(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("Embedder must return a non-empty 2D matrix")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedder returned a zero-length vector")
        return np.ascontiguousarray(matrix / norms, dtype=np.float32)

    def _load(self) -> None:
        assert self.persist_directory is not None
        chunks_path = self.persist_directory / "chunks.json"
        vectors_path = self.persist_directory / "vectors.npz"
        if not chunks_path.exists() and not vectors_path.exists():
            return
        if not chunks_path.exists() or not vectors_path.exists():
            raise NarrativeStoreError("FAISS persistence files are incomplete")
        try:
            serialized_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            with np.load(vectors_path, allow_pickle=False) as archive:
                ids = archive["ids"].tolist()
                vectors = archive["vectors"]
            if len(ids) != len(serialized_chunks) or len(ids) != len(vectors):
                raise ValueError("Persisted chunk and vector counts do not match")
            self._chunks = {
                chunk_id: DocumentChunk.model_validate(data)
                for chunk_id, data in zip(ids, serialized_chunks, strict=True)
            }
            normalized = self._normalized(vectors)
            self._vectors = {
                chunk_id: normalized[position].copy()
                for position, chunk_id in enumerate(ids)
            }
        except Exception as exc:
            raise NarrativeStoreError("Could not load persisted FAISS narrative data") from exc

    def _persist(self) -> None:
        if self.persist_directory is None:
            return
        try:
            ids = list(self._chunks)
            chunks_path = self.persist_directory / "chunks.json"
            vectors_path = self.persist_directory / "vectors.npz"
            chunks_path.write_text(
                json.dumps([self._chunks[chunk_id].model_dump(mode="json") for chunk_id in ids]),
                encoding="utf-8",
            )
            vectors = np.stack([self._vectors[chunk_id] for chunk_id in ids])
            np.savez_compressed(vectors_path, ids=np.asarray(ids), vectors=vectors)
        except Exception as exc:
            raise NarrativeStoreError("Could not persist FAISS narrative data") from exc
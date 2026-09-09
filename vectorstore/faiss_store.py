import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np


class FAISSVectorStore:
    """
    FAISS vector store with persistent index and chunk metadata.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Dict[str, Any]] = []

    def add(
        self,
        embeddings,
        chunks: List[Dict[str, Any]],
    ):
        """Add embeddings and their metadata to FAISS."""

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {embeddings.shape[1]}."
            )

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks."
            )

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search for the most similar chunks."""

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index == -1:
                continue

            result = self.chunks[index].copy()
            result["score"] = float(score)

            results.append(result)

        return results

    def save(
        self,
        index_path: str = "data/faiss.index",
        metadata_path: str = "data/faiss_metadata.json",
    ):
        """Save the FAISS index and chunk metadata."""

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(
        cls,
        index_path: str = "data/faiss.index",
        metadata_path: str = "data/faiss_metadata.json",
    ):
        """Load a previously saved FAISS index and metadata."""

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"FAISS metadata not found: {metadata_path}"
            )

        index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            chunks = json.load(file)

        store = cls(index.d)

        store.index = index
        store.chunks = chunks

        return store
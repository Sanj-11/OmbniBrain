import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add(self, embeddings, chunks):
        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding, top_k=3):

        query_embedding = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            result = self.chunks[index].copy()
            result["score"] = float(score)

            results.append(result)

        return results
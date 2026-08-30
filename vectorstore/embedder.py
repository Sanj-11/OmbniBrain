from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True
        )

    def embed_query(self, query):
        return self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]
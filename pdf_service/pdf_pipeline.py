from pathlib import Path
from typing import Dict, Any

from ingestion.pdf_ingestion import extract_pdf, create_chunks
from vectorstore.embedder import TextEmbedder
from vectorstore.faiss_store import FAISSVectorStore


class PDFPipeline:
    """
    Dataset-specific end-to-end PDF processing and retrieval pipeline.

    Each dataset gets its own FAISS index and metadata file.

    Flow:
        PDF
        -> text extraction
        -> chunking
        -> text embeddings
        -> dataset-specific FAISS index
        -> persistent storage
    """

    def __init__(
        self,
        dataset_id: int,
        base_dir: str = "data/faiss",
    ):
        self.dataset_id = dataset_id

        self.dataset_dir = (
            Path(base_dir) / f"dataset_{dataset_id}"
        )

        self.index_path = (
            self.dataset_dir / "faiss.index"
        )

        self.metadata_path = (
            self.dataset_dir / "faiss_metadata.json"
        )

        # all-MiniLM-L6-v2 produces 384-dimensional embeddings.
        self.embedder = TextEmbedder()

    def process_pdf(
        self,
        pdf_path: str,
    ) -> Dict[str, Any]:
        """
        Process a PDF and create a FAISS index for this dataset.
        """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        # ---------------------------------------------------------
        # 1. Extract PDF
        # ---------------------------------------------------------

        document = extract_pdf(
            str(pdf_path)
        )

        # ---------------------------------------------------------
        # 2. Create text chunks
        # ---------------------------------------------------------

        chunks = create_chunks(
            document
        )

        if not chunks:
            raise ValueError(
                "No text content could be extracted from the PDF."
            )

        # ---------------------------------------------------------
        # 3. Create embeddings
        # ---------------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedder.embed_documents(
            texts
        )

        # ---------------------------------------------------------
        # 4. Create dataset-specific FAISS store
        # ---------------------------------------------------------

        store = FAISSVectorStore(
            dimension=embeddings.shape[1]
        )

        # ---------------------------------------------------------
        # 5. Add embeddings + metadata
        # ---------------------------------------------------------

        store.add(
            embeddings,
            chunks
        )

        # ---------------------------------------------------------
        # 6. Save dataset-specific index
        # ---------------------------------------------------------

        store.save(
            index_path=str(self.index_path),
            metadata_path=str(self.metadata_path),
        )

        # ---------------------------------------------------------
        # 7. Return processing information
        # ---------------------------------------------------------

        return {
            "dataset_id": self.dataset_id,
            "document": document["document"],
            "pages": len(document["pages"]),
            "chunks": len(chunks),
            "embedding_dimension": embeddings.shape[1],
            "vectors": store.index.ntotal,
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
        }

    def search(
        self,
        question: str,
        top_k: int = 3,
    ):
        """
        Search this dataset's persisted FAISS index.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found for dataset "
                f"{self.dataset_id}: {self.index_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"FAISS metadata not found for dataset "
                f"{self.dataset_id}: {self.metadata_path}"
            )

        # ---------------------------------------------------------
        # 1. Load this dataset's FAISS store
        # ---------------------------------------------------------

        store = FAISSVectorStore.load(
            index_path=str(self.index_path),
            metadata_path=str(self.metadata_path),
        )

        # ---------------------------------------------------------
        # 2. Embed the question
        # ---------------------------------------------------------

        query_embedding = self.embedder.embed_query(
            question.strip()
        )

        # ---------------------------------------------------------
        # 3. Semantic search
        # ---------------------------------------------------------

        return store.search(
            query_embedding,
            top_k=top_k,
        )
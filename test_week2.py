from ingestion.pdf_ingestion import extract_pdf, create_chunks
from vectorstore.embedder import TextEmbedder
from vectorstore.faiss_store import FAISSVectorStore


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PDF_PATH = "uploads/OmniBrain_Sample_Financial_Report.pdf"

INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/faiss_metadata.json"


# ---------------------------------------------------------
# 1. Extract PDF
# ---------------------------------------------------------

print("\n[1] PDF INGESTION")
print("=" * 60)

document = extract_pdf(PDF_PATH)

print("Document:", document["document"])
print("Pages:", len(document["pages"]))


# ---------------------------------------------------------
# 2. Create text chunks
# ---------------------------------------------------------

print("\n[2] CHUNKING")
print("=" * 60)

chunks = create_chunks(document)

print("Chunks:", len(chunks))

for chunk in chunks[:2]:
    print(
        f"Chunk ID: {chunk['chunk_id']} | "
        f"Page: {chunk['page']}"
    )


# ---------------------------------------------------------
# 3. Create embeddings
# ---------------------------------------------------------

print("\n[3] EMBEDDINGS")
print("=" * 60)

embedder = TextEmbedder()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedder.embed_documents(texts)

print("Embedding dimension:", embeddings.shape[1])
print("Number of embeddings:", len(embeddings))


# ---------------------------------------------------------
# 4. Create FAISS vector store
# ---------------------------------------------------------

print("\n[4] FAISS VECTOR STORE")
print("=" * 60)

store = FAISSVectorStore(
    dimension=embeddings.shape[1]
)

store.add(
    embeddings,
    chunks
)

print("FAISS vectors:", store.index.ntotal)


# ---------------------------------------------------------
# 5. Save FAISS index + metadata
# ---------------------------------------------------------

print("\n[5] PERSISTENCE")
print("=" * 60)

store.save(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)

print("FAISS index saved to:", INDEX_PATH)
print("Metadata saved to:", METADATA_PATH)


# ---------------------------------------------------------
# 6. Semantic search
# ---------------------------------------------------------

print("\n[6] SEMANTIC SEARCH")
print("=" * 60)

question = "What was the revenue in FY2024?"

print("Question:", question)

query_embedding = embedder.embed_query(
    question
)

results = store.search(
    query_embedding,
    top_k=3
)

print("\nSEARCH RESULTS")
print("-" * 60)

for result in results:

    print(
        f"Score: {result['score']:.4f} | "
        f"Page: {result['page']}"
    )

    print(
        f"Document: {result['document']}"
    )

    print(
        "Text:",
        result["text"][:500]
    )

    print("-" * 60)


# ---------------------------------------------------------
# 7. Load the saved FAISS index
# ---------------------------------------------------------

print("\n[7] LOAD PERSISTED INDEX")
print("=" * 60)

loaded_store = FAISSVectorStore.load(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)

print(
    "Loaded FAISS vectors:",
    loaded_store.index.ntotal
)

print(
    "Loaded metadata chunks:",
    len(loaded_store.chunks)
)


# ---------------------------------------------------------
# 8. Search the loaded index
# ---------------------------------------------------------

print("\n[8] SEARCH LOADED INDEX")
print("=" * 60)

loaded_results = loaded_store.search(
    query_embedding,
    top_k=3
)

for result in loaded_results:

    print(
        f"Score: {result['score']:.4f} | "
        f"Page: {result['page']}"
    )

    print(
        f"Document: {result['document']}"
    )

    print(
        "Text:",
        result["text"][:300]
    )

    print("-" * 60)


# ---------------------------------------------------------
# Final status
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("WEEK 2 PIPELINE TEST COMPLETED SUCCESSFULLY")
print("=" * 60)

print("PDF ingestion       : OK")
print("Chunking             : OK")
print("Embeddings           : OK")
print("FAISS indexing       : OK")
print("FAISS persistence    : OK")
print("Semantic search      : OK")
print("Reload + search      : OK")
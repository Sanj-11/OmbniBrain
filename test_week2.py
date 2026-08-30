from ingestion.pdf_ingestion import extract_pdf, create_chunks
from vectorstore.embedder import TextEmbedder
from vectorstore.faiss_store import FAISSVectorStore


PDF_PATH = "uploads/OmniBrain_Sample_Financial_Report.pdf"


# 1. Extract PDF
document = extract_pdf(PDF_PATH)

# 2. Create chunks
chunks = create_chunks(document)

print("Pages:", len(document["pages"]))
print("Chunks:", len(chunks))


# 3. Create embeddings
embedder = TextEmbedder()

texts = [chunk["text"] for chunk in chunks]

embeddings = embedder.embed_documents(texts)

print("Embedding dimension:", embeddings.shape[1])


# 4. Create FAISS store
store = FAISSVectorStore(
    dimension=embeddings.shape[1]
)

store.add(embeddings, chunks)

print("FAISS vectors:", store.index.ntotal)


# 5. Semantic search
question = "What was the revenue in FY2024?"

query_embedding = embedder.embed_query(question)

results = store.search(
    query_embedding,
    top_k=3
)


print("\nSEARCH RESULTS")
print("=" * 60)

for result in results:

    print(
        f"Score: {result['score']:.4f} | "
        f"Page: {result['page']}"
    )

    print(result["text"][:500])
    print("-" * 60)
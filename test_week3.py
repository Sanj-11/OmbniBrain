from pdf_service.pdf_pipeline import PDFPipeline


print("\n[1] INITIALIZING PDF PIPELINE")
print("=" * 60)

pipeline = PDFPipeline()

print("Pipeline initialized successfully.")


print("\n[2] SEMANTIC SEARCH")
print("=" * 60)

question = "What was the revenue in FY2024?"

print("Question:", question)

results = pipeline.search(
    question,
    top_k=3
)


print("\nSEARCH RESULTS")
print("-" * 60)

for result in results:

    print(
        f"Page: {result['page']} | "
        f"Score: {result['score']:.4f}"
    )

    print(
        "Document:",
        result["document"]
    )

    print(
        "Text:",
        result["text"][:300]
    )

    print("-" * 60)


print("\n" + "=" * 60)
print("WEEK 3 PDF SEARCH TEST COMPLETED")
print("=" * 60)
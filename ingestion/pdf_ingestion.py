from pathlib import Path
from typing import List, Dict, Any

import pymupdf


def extract_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text, images, and page metadata from a PDF.

    Embedded images are extracted when available. Each page is also
    rendered to a PNG so vector-based charts and figures can be
    represented as images for downstream vision/CLIP processing.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        # Extract page text
        text = page.get_text("text").strip()

        images = []

        # Extract embedded raster images
        for image_number, image in enumerate(
            page.get_images(full=True),
            start=1
        ):
            xref = image[0]
            image_data = document.extract_image(xref)

            images.append({
                "image_number": image_number,
                "type": "embedded",
                "xref": xref,
                "extension": image_data["ext"],
                "width": image_data["width"],
                "height": image_data["height"],
            })

        # Render the complete page as an image.
        # This also captures vector-based charts and figures.
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(1.5, 1.5)
        )

        rendered_image = pixmap.tobytes("png")

        images.append({
            "image_number": len(images) + 1,
            "type": "page_render",
            "extension": "png",
            "width": pixmap.width,
            "height": pixmap.height,
            "data": rendered_image,
        })

        pages.append({
            "page": page_number,
            "text": text,
            "images": images,
        })

    document.close()

    return {
        "document": pdf_path.name,
        "pages": pages,
    }


def create_chunks(
    extracted_document: Dict[str, Any],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Split extracted PDF text into overlapping chunks while
    preserving document and page metadata.
    """

    chunks = []

    for page_data in extracted_document["pages"]:

        text = page_data["text"]

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": len(chunks),
                    "document": extracted_document["document"],
                    "page": page_data["page"],
                    "text": chunk_text,
                })

            if end >= len(text):
                break

            start = end - overlap

    return chunks
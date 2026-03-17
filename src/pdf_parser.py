"""
pdf_parser.py
-------------
Extracts text blocks and images from each page of a PDF using PyMuPDF.
Returns a list of PageData objects ready for the extraction pipeline.
"""

import fitz  
import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image


@dataclass
class PageData:
    page_number: int
    raw_text: str
    images: list = field(default_factory=list)   # list of PIL.Image
    page_image: Image.Image = None               # full-page render


def pdf_to_pages(pdf_path: str, dpi: int = 150) -> list[PageData]:
    """
    Parse every page of the PDF.

    Args:
        pdf_path: Path to the PDF file.
        dpi:      Resolution for rendering page images (higher = better quality but slower).

    Returns:
        List of PageData, one per page.
    """
    pages = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        # ── Text extraction ────────────────────────────────────────────────
        raw_text = page.get_text("text").strip()

        # ── Embedded image extraction ──────────────────────────────────────
        embedded_images = []
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                embedded_images.append(pil_img)
            except Exception:
                pass  # skip corrupt images

        # ── Full-page render ───────────────────────────────────────────────
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        page_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        pages.append(
            PageData(
                page_number=page_num,
                raw_text=raw_text,
                images=embedded_images,
                page_image=page_pil,
            )
        )

    doc.close()
    return pages


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL image to a base64 string (for display in HTML/Streamlit)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

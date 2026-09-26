import fitz
from pathlib import Path

try:
    import ocrmypdf
except ImportError:
    ocrmypdf = None


def extract_text_from_pdf(pdf_path, ocr_output="ocr_output.pdf"):
    """
    Extract text using PyMuPDF.
    If the PDF is scanned/image-based, use OCRmyPDF or fallback.
    """

    # -------------------------------
    # STEP 1: Try PyMuPDF extraction
    # -------------------------------
    doc = fitz.open(pdf_path)

    pages_text = []

    for page in doc:
        text = page.get_text("text")
        pages_text.append(text.strip())

    doc.close()

    extracted_text = "\n\n".join(pages_text)

    # -------------------------------
    # STEP 2: Check if OCR is needed
    # -------------------------------
    if len(extracted_text.strip()) > 100:
        print("Text-based PDF detected.")
        return extracted_text

    # -------------------------------
    # STEP 3: OCR fallback
    # -------------------------------
    print("Scanned PDF detected.")
    if ocrmypdf is not None:
        print("Running OCR with OCRmyPDF...")
        ocrmypdf.ocr(
            str(pdf_path),
            str(ocr_output),
            deskew=True,
            force_ocr=True
        )
        doc = fitz.open(ocr_output)
        pages_text = [page.get_text("text").strip() for page in doc]
        doc.close()
        return "\n\n".join(pages_text)
    else:
        # Check if pre-existing ocr_output exists
        ocr_file = Path(ocr_output)
        if ocr_file.exists():
            print(f"Using pre-existing OCR output: {ocr_output}")
            doc = fitz.open(str(ocr_file))
            pages_text = [page.get_text("text").strip() for page in doc]
            doc.close()
            return "\n\n".join(pages_text)
        print("OCRmyPDF is not installed. Returning extracted text so far.")
        return extracted_text


# ---------------------------------
# TEST
# ---------------------------------

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    pdf_path = BASE_DIR / "sample_contract.pdf"
    if not pdf_path.exists():
        pdf_path = BASE_DIR / "ocr_output.pdf"
    if not pdf_path.exists():
        pdf_path = BASE_DIR / "scanned_copy.pdf"

    print(f"Testing extraction from: {pdf_path}")
    text = extract_text_from_pdf(str(pdf_path))

    print("\n========== EXTRACTED TEXT (First 500 chars) ==========\n")
    print(text[:500])
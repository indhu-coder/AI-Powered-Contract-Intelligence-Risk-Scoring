import fitz
import ocrmypdf
import os


def extract_text_from_pdf(pdf_path, ocr_output="ocr_output.pdf"):
    """
    Extract text using PyMuPDF.
    If the PDF is scanned/image-based, use OCRmyPDF.
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
    print("Running OCR...")

    ocrmypdf.ocr(
        pdf_path,
        ocr_output,
        deskew=True,
        force_ocr=True
    )

    # -------------------------------
    # STEP 4: Extract OCR text
    # -------------------------------
    doc = fitz.open(ocr_output)

    pages_text = []

    for page in doc:
        text = page.get_text("text")
        pages_text.append(text.strip())

    doc.close()

    extracted_text = "\n\n".join(pages_text)

    return extracted_text


# ---------------------------------
# TEST
# ---------------------------------

pdf_path = r"D:\\ContractIntelligence\\scanned_copy.pdf"

text = extract_text_from_pdf(pdf_path)

print("\n========== EXTRACTED TEXT ==========\n")
print(text[:500])  # Print first 500 characters
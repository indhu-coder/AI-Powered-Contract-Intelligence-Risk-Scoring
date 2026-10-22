import os
import re
import tempfile
from pathlib import Path

import fitz
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Uses the existing saved model. No retraining is performed.\
MODEL_PATH = r"D:\\ContractIntelligence\\ContractIQ-main\\ContractIQ-main\\ml\\models\\final"
MAX_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_TOKENS = 32

# st.set_page_config(page_title="Contract Intelligence QA", layout="wide")
# st.title("AI-Powered Contract Intelligence")
# st.caption("Upload a contract PDF and ask a question about its clauses.")


@st.cache_resource
# def load_model(model_path):
#     if not os.path.isdir(model_path):
#         raise FileNotFoundError(
#             f"Saved model folder not found: {model_path}. Check MODEL_PATH at the top."
#         )
#     tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
#     model = AutoModelForQuestionAnswering.from_pretrained(
#         model_path, local_files_only=True
#     )
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#     return tokenizer, model, device


# def extract_pdf_text(uploaded_file):
#     pdf_bytes = uploaded_file.getvalue()
#     try:
#         with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
#             text = "\n\n".join(page.get_text("text").strip() for page in doc)
#     except Exception as exc:
#         raise ValueError(f"Could not read PDF: {exc}") from exc

#     if len(text.strip()) < 100:
#         try:
#             import ocrmypdf
#             with tempfile.TemporaryDirectory() as tmp:
#                 src, dst = Path(tmp) / "source.pdf", Path(tmp) / "ocr.pdf"
#                 src.write_bytes(pdf_bytes)
#                 ocrmypdf.ocr(str(src), str(dst), deskew=True, force_ocr=True)
#                 with fitz.open(str(dst)) as doc:
#                     text = "\n\n".join(page.get_text("text").strip() for page in doc)
#         except ImportError:
#             st.warning("This may be a scanned PDF. Install OCRmyPDF and Tesseract for OCR.")
#         except Exception as exc:
#             st.warning(f"OCR fallback failed: {exc}")

#     text = re.sub(r"[ \t]+", " ", text)
#     text = re.sub(r"\n{3,}", "\n\n", text).strip()
#     if not text:
#         raise ValueError("No readable text was extracted from this PDF.")
#     return text


def split_sections(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections = []
    for paragraph in paragraphs:
        if len(paragraph) <= 4000:
            sections.append(paragraph)
        else:
            # Overlapping windows preserve context in long paragraphs.
            step = 3000
            sections.extend(paragraph[i:i + 3500].strip() for i in range(0, len(paragraph), step))
    return [s for s in sections if s] or [text]


STOPWORDS = {
    "what", "when", "where", "who", "which", "how", "is", "are", "was", "were",
    "the", "a", "an", "of", "to", "for", "in", "on", "does", "do", "did", "this",
    "that", "under", "with", "and", "or", "by", "be", "it", "their", "its",
    "agreement", "please", "tell", "me"
}


def retrieve_sections(question, sections, limit=10):
    q = question.lower()
    terms = set(re.findall(r"[a-z]{3,}", q)) - STOPWORDS
    if any(x in q for x in ("notice", "terminate", "termination", "cancellation")):
        terms.update({"notice", "termination", "terminate", "terminated", "cancel"})
    if "effective date" in q or "commencement date" in q:
        terms.update({"effective", "commencement", "date", "dated"})
    if any(x in q for x in ("expiration date", "expiry date", "end date")):
        terms.update({"expiration", "expiry", "expires", "expire", "ends"})
    if "renewal term" in q:
        terms.update({"renewal", "renew", "term"})

    ranked = []
    for idx, section in enumerate(sections):
        low = section.lower()
        score = 2 * sum(bool(re.search(rf"\b{re.escape(t)}\b", low)) for t in terms if t not in {"terminate", "cancel"})
        if re.search(r"\b(terminat\w*|cancel\w*)\b", low) and any(x in q for x in ("terminate", "termination", "cancellation")):
            score += 6
        if "notice" in q and re.search(r"\bnotice\b", low):
            score += 8
        if any(x in q for x in ("notice", "terminate", "termination")) and re.search(r"\b(days?|weeks?|months?|years?)\b", low):
            score += 2
        if "effective date" in q and "effective" in low:
            score += 10
        if any(x in q for x in ("expiration date", "expiry date", "end date")) and re.search(r"\b(expir\w*|ends on|ending on)\b", low):
            score += 8
        ranked.append((score, idx, section))

    selected = sorted((x for x in ranked if x[0] > 0), key=lambda x: (-x[0], x[1]))[:limit]
    if not selected:
        return sections[:limit]
    selected.sort(key=lambda x: x[1])
    return [x[2] for x in selected]


NUMBER_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
DURATION_RE = re.compile(
    rf"\b(?:\d+|{NUMBER_WORDS})(?:\s*\(\s*\d+\s*\))?\s+"
    r"(?:calendar\s+|business\s+)?(?:days?|weeks?|months?|years?)\b", re.I
)


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if s.strip()]


def targeted_duration_answer(question, context):
    q = question.lower()
    if not any(x in q for x in ("notice", "terminate", "termination", "renewal term", "how long")):
        return None
    candidates = []
    for sentence in split_sentences(context):
        low = sentence.lower()
        if not re.search(r"\b(notice|terminat\w*|cancel\w*|renewal)\b", low):
            continue
        match = DURATION_RE.search(sentence)
        if not match:
            continue
        if "notice" in q and "notice" not in low:
            continue
        # Return the relevant contractual sentence, preserving words such as "prior notice".
        candidates.append((sentence, bool(re.search(r"\bnotice\b", low)) + bool(re.search(r"\bterminat\w*\b", low))))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1], -len(x[0])), reverse=True)
    sentence = candidates[0][0]
    return {"answer": sentence, "evidence": sentence, "source": "Relevant notice/termination clause"}


def targeted_date_answer(question, context):
    q = question.lower()
    if not any(x in q for x in ("effective date", "commencement date", "expiration date", "expiry date", "agreement date", "execution date")):
        return None
    if "effective" in q or "commencement" in q:
        key = r"\b(effective date|commencement date|becomes effective|shall be effective)\b"
    elif "expiration" in q or "expiry" in q:
        key = r"\b(expiration date|expiry date|expires|shall expire|ends on|ending on)\b"
    else:
        key = r"\b(agreement date|dated as of|executed on|execution date)\b"
    date_re = re.compile(
        r"\b(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b"
        r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}\b"
        r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b", re.I
    )
    for sentence in split_sentences(context):
        if re.search(key, sentence, re.I):
            match = date_re.search(sentence)
            if match:
                return {"answer": match.group(0), "evidence": sentence, "source": "Matching date clause"}
    # Never return an unrelated date as a fallback.
    return None


def bert_answer(question, context, tokenizer, model, device):
    encoded = tokenizer(
        question, context, truncation="only_second", max_length=MAX_LENGTH,
        stride=DOC_STRIDE, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding="max_length", return_tensors="pt"
    )
    offsets = encoded.pop("offset_mapping")
    encoded.pop("overflow_to_sample_mapping", None)
    seq_ids = [encoded.sequence_ids(i) for i in range(offsets.shape[0])]
    model_inputs = {k: v.to(device) for k, v in encoded.items()}
    with torch.inference_mode():
        outputs = model(**model_inputs)

    best = None
    for feature_idx, sequence_ids in enumerate(seq_ids):
        start_logits = outputs.start_logits[feature_idx]
        end_logits = outputs.end_logits[feature_idx]
        feature_offsets = offsets[feature_idx]
        context_positions = [i for i, sid in enumerate(sequence_ids) if sid == 1]
        for start in context_positions:
            for end in range(start, min(start + MAX_ANSWER_TOKENS, len(sequence_ids) - 1) + 1):
                if sequence_ids[end] != 1:
                    break
                char_start = int(feature_offsets[start][0])
                char_end = int(feature_offsets[end][1])
                if char_end <= char_start:
                    continue
                answer = context[char_start:char_end].strip()
                if not answer or len(answer) > 500:
                    continue
                score = float(start_logits[start].item() + end_logits[end].item())
                if best is None or score > best["score"]:
                    best = {
                        "answer": answer, "score": score,
                        "evidence": context[max(0, char_start - 180):min(len(context), char_end + 220)].strip(),
                        "source": "Saved BERT-CUAD model"
                    }
    if best and best["answer"].lower().strip(" .,:;") not in {"the", "this", "that", "and", "for", "with", "shall", "agreement"}:
        return best
    return None


# uploaded_file = st.file_uploader("Upload contract PDF", type=["pdf"])
# question = st.text_input("Enter your question", placeholder="What is the notice period to terminate the agreement?")

if uploaded_file:
    try:
        with st.spinner("Extracting contract text..."):
            contract_text = extract_pdf_text(uploaded_file)
        st.success(f"Extracted {len(contract_text):,} characters.")
        with st.expander("Preview extracted contract text"):
            st.text(contract_text[:5000])

        if question.strip():
            with st.spinner("Finding relevant clauses and answering..."):
                tokenizer, model, device = load_model(MODEL_PATH)
                sections = split_sections(contract_text)
                chosen_sections = retrieve_sections(question, sections)
                relevant_context = "\n\n".join(chosen_sections)
                result = targeted_duration_answer(question, relevant_context)
                if result is None:
                    result = targeted_date_answer(question, relevant_context)
                if result is None:
                    result = bert_answer(question, relevant_context, tokenizer, model, device)

            # if result and result.get("answer", "").strip():
            #     st.subheader("Answer")
            #     st.write(result["answer"])
            #     st.caption(f"Source: {result.get('source', 'QA model')}")
            #     st.subheader("Supporting text")
            #     st.write(result.get("evidence", ""))
            #     st.info("Verify the answer against the original clause. This tool supports contract review and is not legal advice.")
            #     if "score" in result:
            #         with st.expander("Model diagnostics"):
            #             st.write(f"Raw span score: {result['score']:.4f}")
            #             st.write(f"Device: {device}")
            #             st.write(f"Relevant sections selected: {len(chosen_sections)}")
            # else:
#                 st.warning("No reliable answer was found in the relevant clauses. The app has not substituted an unrelated date or duration.")
#                 with st.expander("Relevant clauses checked"):
#                     st.write(relevant_context[:10000])
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
else:
    st.write("Upload a PDF to begin.")

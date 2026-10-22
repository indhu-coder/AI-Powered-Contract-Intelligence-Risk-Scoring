
import os
import re
import tempfile
from pathlib import Path

import fitz
import ocrmypdf
import pandas as pd
import spacy
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(page_title="Contract Intelligence QA", layout="wide")
st.title("AI-Powered Contract Intelligence")

MODEL_PATH = Path(
    r"D:\\ContractIntelligence\\bert-cuad-qa-transformers-default-v4"
)

MAX_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_LENGTH = 30

# Based on your earlier validation threshold sweep.
# Revalidate this threshold on your held-out CUAD test/validation split.
NULL_SCORE_DIFF_THRESHOLD = 27.0


# ============================================================
# LOAD EXISTING MODEL — NO RETRAINING
# ============================================================

@st.cache_resource
def load_qa_model():
    if not (MODEL_PATH / "config.json").exists():
        raise FileNotFoundError(
            f"Model config.json was not found at: {MODEL_PATH}"
        )

    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        str(MODEL_PATH)
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qa_model.to(dev)
    qa_model.eval()

    return tok, qa_model, dev


@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")


try:
    tokenizer, model, device = load_qa_model()
    nlp = load_spacy_model()
    st.caption(f"Model loaded successfully · Device: {device}")
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()


# ============================================================
# PDF TEXT EXTRACTION — LOCAL OCR FALLBACK
# ============================================================

def extract_pdf_text(uploaded_file):
    if uploaded_file is None:
        return ""

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "uploaded.pdf"
        ocr_path = Path(temp_dir) / "ocr_output.pdf"

        input_path.write_bytes(uploaded_file.getvalue())

        try:
            with fitz.open(str(input_path)) as doc:
                pages = [page.get_text("text").strip() for page in doc]

            extracted_text = "\n\n".join(pages).strip()

            if len(extracted_text) > 100:
                return extracted_text

            # Scanned PDF: OCR locally before text extraction.
            ocrmypdf.ocr(
                str(input_path),
                str(ocr_path),
                deskew=True,
                force_ocr=True,
                progress_bar=False
            )

            with fitz.open(str(ocr_path)) as doc:
                pages = [page.get_text("text").strip() for page in doc]

            return "\n\n".join(pages).strip()

        except Exception as exc:
            raise RuntimeError(
                f"PDF extraction/OCR failed: {exc}"
            ) from exc


# ============================================================
# DATE AND DURATION EXTRACTION
# ============================================================

DATE_REGEX = re.compile(
    r"\b(?:"
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}"
    r"|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|"
    r"\d{4}-\d{1,2}-\d{1,2}"
    r")\b",
    re.IGNORECASE
)

DURATION_REGEX = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|"
    r"forty|fifty|sixty|ninety|hundred)"
    r"(?:\s*\(\s*\d+\s*\))?"
    r"\s*[- ]?\s*(?:calendar\s+|business\s+)?"
    r"(?:days?|weeks?|months?|years?)\b",
    re.IGNORECASE
)


def find_relevant_value(question, context, regex, keywords):
    """Find a date/duration near wording relevant to the question."""
    matches = list(regex.finditer(context))
    if not matches:
        return ""

    q = question.lower()

    relevant_keywords = [
        keyword for keyword in keywords
        if keyword in q
    ]

    ranked = []

    for match in matches:
        left = context[max(0, match.start() - 220):match.start()].lower()
        right = context[match.end():min(len(context), match.end() + 120)].lower()
        surrounding = left + " " + right

        score = sum(
            2 if keyword in left else 1
            for keyword in relevant_keywords
            if keyword in surrounding
        )

        ranked.append((score, match.start(), match.group().strip()))

    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    if ranked and ranked[0][0] > 0:
        return ranked[0][2]

    return ""


DATE_KEYWORDS = [
    "effective date", "effective as of",
    "agreement date", "dated as of", "executed on",
    "expiration date", "expiry date", "expires on",
    "termination date", "terminated on",
    "commencement date", "commencement",
    "start date", "beginning on"
]

DURATION_KEYWORDS = [
    "notice period", "notice to terminate",
    "prior written notice", "days' notice",
    "renewal term", "renewal period",
    "initial term", "term of this agreement",
    "termination notice", "cure period",
    "extension period", "duration"
]


# ============================================================
# BERT QUESTION ANSWERING — ALL CHUNKS
# ============================================================

def predict_answer(question, context):
    question = question.strip()
    context = context.strip()

    if not question or not context:
        return {
            "answer": "",
            "candidate": "",
            "candidate_score": float("-inf"),
            "null_score": float("inf"),
            "null_minus_candidate": float("inf")
        }

    encoded = tokenizer(
        question,
        context,
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt"
    )

    offsets = encoded["offset_mapping"]
    number_of_chunks = encoded["input_ids"].shape[0]

    best_candidate = ""
    best_candidate_score = float("-inf")
    null_scores = []

    model_keys = {
        "input_ids",
        "attention_mask",
        "token_type_ids"
    }

    for chunk_idx in range(number_of_chunks):
        sequence_ids = encoded.sequence_ids(chunk_idx)
        chunk_offsets = offsets[chunk_idx].tolist()

        chunk_inputs = {
            key: value[chunk_idx:chunk_idx + 1].to(device)
            for key, value in encoded.items()
            if key in model_keys
        }

        with torch.no_grad():
            outputs = model(**chunk_inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # CLS/null score for this chunk
        cls_index = 0
        null_scores.append(
            start_logits[cls_index].item()
            + end_logits[cls_index].item()
        )

        # Restrict candidate spans to context tokens.
        context_indices = [
            i for i, seq_id in enumerate(sequence_ids)
            if seq_id == 1
            and chunk_offsets[i][1] > chunk_offsets[i][0]
        ]

        if not context_indices:
            continue

        start_top = torch.topk(
            start_logits, k=min(20, len(start_logits))
        ).indices.tolist()

        end_top = torch.topk(
            end_logits, k=min(20, len(end_logits))
        ).indices.tolist()

        context_set = set(context_indices)

        for start_idx in start_top:
            if start_idx not in context_set:
                continue

            start_char = chunk_offsets[start_idx][0]

            for end_idx in end_top:
                if end_idx < start_idx:
                    continue

                if end_idx - start_idx + 1 > MAX_ANSWER_LENGTH:
                    continue

                if end_idx not in context_set:
                    continue

                end_char = chunk_offsets[end_idx][1]

                if end_char <= start_char:
                    continue

                candidate_score = (
                    start_logits[start_idx].item()
                    + end_logits[end_idx].item()
                )

                if candidate_score > best_candidate_score:
                    candidate = context[start_char:end_char].strip()

                    if candidate:
                        best_candidate = candidate
                        best_candidate_score = candidate_score

    null_score = min(null_scores) if null_scores else float("inf")

    score_difference = (
        null_score - best_candidate_score
        if best_candidate
        else float("inf")
    )

    # Keep raw candidate for troubleshooting; use the null threshold
    # to decide whether the model should return an empty answer.
    answer = (
        best_candidate
        if best_candidate
        and score_difference <= NULL_SCORE_DIFF_THRESHOLD
        else ""
    )

    # If BERT returns empty, use a relevant date/duration fallback.
    # A duration is never converted into a calendar date.
    q = question.lower()

    if not answer and any(
        term in q for term in [
            "effective date", "agreement date", "expiration date",
            "expiry date", "termination date", "commencement date",
            "start date"
        ]
    ):
        answer = find_relevant_value(
            question, context, DATE_REGEX, DATE_KEYWORDS
        )

    elif not answer and any(
        term in q for term in [
            "notice period", "renewal term", "renewal period",
            "initial term", "cure period", "duration",
            "how many days", "how many months", "how many years"
        ]
    ):
        answer = find_relevant_value(
            question, context, DURATION_REGEX, DURATION_KEYWORDS
        )

    return {
        "answer": answer,
        "candidate": best_candidate,
        "candidate_score": best_candidate_score,
        "null_score": null_score,
        "null_minus_candidate": score_difference
    }


# ============================================================
# LEGAL ENTITY EXTRACTION
# ============================================================

LEGAL_TERMS = {
    "termination": "TERMINATION",
    "notice period": "NOTICE_PERIOD",
    "effective date": "EFFECTIVE_DATE",
    "agreement date": "AGREEMENT_DATE",
    "expiration": "EXPIRATION",
    "renewal": "RENEWAL",
    "confidentiality": "CONFIDENTIALITY",
    "non-disclosure": "CONFIDENTIALITY",
    "governing law": "GOVERNING_LAW",
    "jurisdiction": "JURISDICTION",
    "indemnification": "INDEMNIFICATION",
    "indemnity": "INDEMNIFICATION",
    "liability": "LIABILITY",
    "warranty": "WARRANTY",
    "assignment": "ASSIGNMENT",
    "intellectual property": "INTELLECTUAL_PROPERTY",
    "force majeure": "FORCE_MAJEURE",
    "arbitration": "DISPUTE_RESOLUTION",
    "dispute resolution": "DISPUTE_RESOLUTION",
    "payment": "PAYMENT",
    "non-compete": "NON_COMPETE"
}


def extract_legal_entities(context):
    doc = nlp(context[:nlp.max_length - 1])
    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        }
        for ent in doc.ents
        if ent.label_ in {
            "ORG", "PERSON", "GPE", "LOC", "DATE",
            "MONEY", "LAW"
        }
    ]

    for match in DATE_REGEX.finditer(context):
        entities.append({
            "text": match.group(),
            "label": "EXACT_DATE",
            "start": match.start(),
            "end": match.end()
        })

    for match in DURATION_REGEX.finditer(context):
        entities.append({
            "text": match.group(),
            "label": "DURATION",
            "start": match.start(),
            "end": match.end()
        })

    lowered_context = context.lower()

    for term, label in LEGAL_TERMS.items():
        for match in re.finditer(re.escape(term), lowered_context):
            entities.append({
                "text": context[match.start():match.end()],
                "label": label,
                "start": match.start(),
                "end": match.end()
            })

    return sorted(
        entities,
        key=lambda item: (item["start"], item["end"])
    )


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a contract PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    try:
        with st.spinner("Extracting contract text..."):
            contract_text = extract_pdf_text(uploaded_file)

        if not contract_text:
            st.error("No text could be extracted from this PDF.")
            st.stop()

        st.success(
            f"Contract extracted successfully · "
            f"{len(contract_text):,} characters"
        )

        with st.expander("Preview extracted contract"):
            st.text(contract_text[:5000])

        with st.expander("Extracted legal entities"):
            entities = extract_legal_entities(contract_text)

            if entities:
                entities_df = pd.DataFrame(entities)
                st.dataframe(entities_df, use_container_width=True)

                grouped = (
                    entities_df.groupby("label")["text"]
                    .apply(lambda values: ", ".join(dict.fromkeys(values)))
                    .reset_index()
                )
                st.subheader("Entities grouped by label")
                st.dataframe(grouped, use_container_width=True)
            else:
                st.info("No legal entities were detected.")

        st.subheader("Ask a question about the contract")

        question = st.text_input(
            "Enter your question",
            placeholder=(
                "Example: What is the notice period to terminate "
                "this agreement?"
            )
        )

        if st.button("Get Answer", type="primary", disabled=not question.strip()):
            with st.spinner("Analyzing the contract with BERT..."):
                result = predict_answer(question, contract_text)

            st.subheader("Answer")
            if result["answer"]:
                st.write(result["answer"])
            else:
                st.warning(
                    "The model did not identify a reliable answer. "
                    "Check the contract wording and the question."
                )

            with st.expander("Inference diagnostics"):
                st.write("Raw BERT candidate:", result["candidate"])
                st.write("Candidate score:", result["candidate_score"])
                st.write("Null score:", result["null_score"])
                st.write(
                    "Null minus candidate:",
                    result["null_minus_candidate"]
                )
                st.write(
                    "Null threshold:",
                    NULL_SCORE_DIFF_THRESHOLD
                )

    except Exception as exc:
        st.error(f"An error occurred: {exc}")

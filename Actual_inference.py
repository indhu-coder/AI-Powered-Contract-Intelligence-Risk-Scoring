import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import streamlit as st
import fitz  # PyMuPDF
import ocrmypdf
import spacy
import pandas as pd
st.title("Contract Intelligence QA System")
# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = r"D:\\ContractIntelligence\\bert-cuad-download"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded successfully!")
print("Device:", device)

# ============================================================
# OCR + CONTEXT
# ============================================================
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

pdf_path = st.file_uploader("Upload a file", type=["pdf","jpeg","png"])
question = st.text_input("Enter your question:")
text = extract_text_from_pdf(pdf_path)  # Use the same PDF file for context

st.write("Extracted Text: " + text[:500] + "...")  # Display first 500 characters of the extracted text

# ============================================================
# LOAD SPACY MODEL
# ============================================================

nlp = spacy.load("en_core_web_sm")

print("SpaCy NER model loaded successfully.")


# ============================================================
# ENTITY RULER
# ============================================================

ruler = nlp.add_pipe(
    "entity_ruler",
    before="ner"
)


# ============================================================
# LEGAL REGEX PATTERNS
# ============================================================

patterns = [

    # -------------------------------
    # TERMINATION
    # -------------------------------
    {
        "label": "TERMINATION",
        "pattern": [
            {"LOWER": {"REGEX": "terminat(e|ed|ion|ing)"}}
        ]
    },

    # -------------------------------
    # NOTICE PERIOD
    # -------------------------------
    {
        "label": "NOTICE_PERIOD",
        "pattern": [
            {"LOWER": {"REGEX": "notice"}},
            {"LOWER": {"REGEX": "period|requirement"}}
        ]
    },

    {
        "label": "NOTICE_PERIOD",
        "pattern": [
            {"LOWER": {"REGEX": "notice"}},
            {"LOWER": {"REGEX": "to"}},
            {"LOWER": {"REGEX": "terminat(e|ion|ing|ed)"}}
        ]
    },

    {
        "label": "NOTICE_PERIOD",
        "pattern": [
            {"LIKE_NUM": True},
            {"LOWER": {"REGEX": "day|days|month|months|year|years"}},
            {"LOWER": {"REGEX": "notice"}}
        ]
    },

    # -------------------------------
    # EFFECTIVE DATE
    # -------------------------------
    {
        "label": "EFFECTIVE_DATE",
        "pattern": [
            {"LOWER": {"REGEX": "effective"}},
            {"LOWER": {"REGEX": "date|from|as"}}
        ]
    },

    {
        "label": "EFFECTIVE_DATE",
        "pattern": [
            {"LOWER": "shall"},
            {"LOWER": {"REGEX": "become|remain"}},
            {"LOWER": "effective"}
        ]
    },

    # -------------------------------
    # AGREEMENT DATE
    # -------------------------------
    {
        "label": "AGREEMENT_DATE",
        "pattern": [
            {"LOWER": "agreement"},
            {"LOWER": "date"}
        ]
    },

    {
        "label": "AGREEMENT_DATE",
        "pattern": [
            {"LOWER": {"REGEX": "dated|executed"}},
            {"LOWER": {"REGEX": "as|on"}}
        ]
    },

    # -------------------------------
    # EXPIRATION
    # -------------------------------
    {
        "label": "EXPIRATION",
        "pattern": [
            {"LOWER": {"REGEX": "expir(e|ation|ed|ing)"}}
        ]
    },

    # -------------------------------
    # RENEWAL
    # -------------------------------
    {
        "label": "RENEWAL",
        "pattern": [
            {"LOWER": {"REGEX": "renew(al|ed|ing)?"}},
            {"LOWER": {"REGEX": "term|period|date"}}
        ]
    },

    # -------------------------------
    # CONFIDENTIALITY
    # -------------------------------
    {
        "label": "CONFIDENTIALITY",
        "pattern": [
            {"LOWER": {"REGEX": "confidential|confidentiality"}}
        ]
    },

    {
        "label": "CONFIDENTIALITY",
        "pattern": [
            {"LOWER": {"REGEX": "non[- ]?disclosure|nondisclosure"}}
        ]
    },

    # -------------------------------
    # GOVERNING LAW
    # -------------------------------
    {
        "label": "GOVERNING_LAW",
        "pattern": [
            {"LOWER": "governing"},
            {"LOWER": "law"}
        ]
    },

    {
        "label": "GOVERNING_LAW",
        "pattern": [
            {"LOWER": {"REGEX": "governed"}},
            {"LOWER": "by"},
            {"LOWER": "the"},
            {"LOWER": {"REGEX": "laws|law"}}
        ]
    },

    # -------------------------------
    # JURISDICTION
    # -------------------------------
    {
        "label": "JURISDICTION",
        "pattern": [
            {"LOWER": {"REGEX": "jurisdiction|jurisdictions"}}
        ]
    },

    # -------------------------------
    # INDEMNIFICATION
    # -------------------------------
    {
        "label": "INDEMNIFICATION",
        "pattern": [
            {"LOWER": {"REGEX": "indemnif(y|ies|ied|ying)|indemnification|indemnity"}}
        ]
    },

    # -------------------------------
    # LIABILITY
    # -------------------------------
    {
        "label": "LIABILITY",
        "pattern": [
            {"LOWER": {"REGEX": "liabilit(y|ies)"}}
        ]
    },

    {
        "label": "LIABILITY",
        "pattern": [
            {"LOWER": "limitation"},
            {"LOWER": "of"},
            {"LOWER": "liability"}
        ]
    },

    # -------------------------------
    # WARRANTY
    # -------------------------------
    {
        "label": "WARRANTY",
        "pattern": [
            {"LOWER": {"REGEX": "warrant(y|ies|ed|ing)"}}
        ]
    },

    # -------------------------------
    # ASSIGNMENT
    # -------------------------------
    {
        "label": "ASSIGNMENT",
        "pattern": [
            {"LOWER": {"REGEX": "assign(ment|ed|ing)?|assignment"}}
        ]
    },

    # -------------------------------
    # INTELLECTUAL PROPERTY
    # -------------------------------
    {
        "label": "INTELLECTUAL_PROPERTY",
        "pattern": [
            {"LOWER": "intellectual"},
            {"LOWER": "property"}
        ]
    },

    {
        "label": "INTELLECTUAL_PROPERTY",
        "pattern": [
            {"LOWER": "intellectual"},
            {"LOWER": "property"},
            {"LOWER": "rights"}
        ]
    },

    # -------------------------------
    # FORCE MAJEURE
    # -------------------------------
    {
        "label": "FORCE_MAJEURE",
        "pattern": [
            {"LOWER": "force"},
            {"LOWER": "majeure"}
        ]
    },

    # -------------------------------
    # DISPUTE RESOLUTION
    # -------------------------------
    {
        "label": "DISPUTE_RESOLUTION",
        "pattern": [
            {"LOWER": "dispute"},
            {"LOWER": {"REGEX": "resolution|settlement"}}
        ]
    },

    {
        "label": "DISPUTE_RESOLUTION",
        "pattern": [
            {"LOWER": {"REGEX": "arbitration|arbitral"}}
        ]
    },

    # -------------------------------
    # PAYMENT
    # -------------------------------
    {
        "label": "PAYMENT",
        "pattern": [
            {"LOWER": {"REGEX": "payment|payments"}}
        ]
    },

    # -------------------------------
    # NON-COMPETE
    # -------------------------------
    {
        "label": "NON_COMPETE",
        "pattern": [
            {"LOWER": {"REGEX": "non[- ]?compete|non[- ]?competition"}}
        ]
    }
]

# ============================================================
# DATE PATTERNS
# ============================================================

date_patterns = [

    # January 31, 2025
    {
        "label": "EXACT_DATE",
        "pattern": [
            {
                "LOWER": {
                    "REGEX": r"january|february|march|april|may|june|"
                              r"july|august|september|october|november|december"
                }
            },
            {
                "LIKE_NUM": True
            },
            {
                "TEXT": ","
            },
            {
                "LIKE_NUM": True
            }
        ]
    },

    # January 31
    {
        "label": "EXACT_DATE",
        "pattern": [
            {
                "LOWER": {
                    "REGEX": r"january|february|march|april|may|june|"
                              r"july|august|september|october|november|december"
                }
            },
            {
                "LIKE_NUM": True
            }
        ]
    },

    # Jan. 31, 2025 / Jan 31 2025
    {
        "label": "EXACT_DATE",
        "pattern": [
            {
                "LOWER": {
                    "REGEX": r"jan\.?|feb\.?|mar\.?|apr\.?|may|jun\.?|"
                              r"jul\.?|aug\.?|sep\.?|sept\.?|oct\.?|"
                              r"nov\.?|dec\.?"
                }
            },
            {
                "LIKE_NUM": True
            },
            {
                "TEXT": ","
            },
            {
                "LIKE_NUM": True
            }
        ]
    },

    # 01/31/2025 or 01-31-2025
    {
        "label": "EXACT_DATE",
        "pattern": [
            {
                "TEXT": {
                    "REGEX": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
                }
            }
        ]
    },

    # 2025-01-31
    {
        "label": "EXACT_DATE",
        "pattern": [
            {
                "TEXT": {
                    "REGEX": r"\d{4}-\d{1,2}-\d{1,2}"
                }
            }
        ]
    }
]


patterns.extend(date_patterns)
# ============================================================
# DURATION / NOTICE PERIOD PATTERNS
# ============================================================

duration_patterns = [

    # 30 days / 30-day / 30 days'
    {
        "label": "DURATION",
        "pattern": [
            {"LIKE_NUM": True},
            {"LOWER": {"REGEX": r"days?|weeks?|months?|years?"}}
        ]
    },

    # thirty days / sixty months / one year
    {
        "label": "DURATION",
        "pattern": [
            {"LOWER": {
                "REGEX": r"(one|two|three|four|five|six|seven|eight|nine|ten|"
                          r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                          r"seventeen|eighteen|nineteen|twenty|thirty|forty|"
                          r"fifty|sixty|seventy|eighty|ninety)"
            }},
            {"LOWER": {"REGEX": r"days?|weeks?|months?|years?"}}
        ]
    },

    # 30-day
    {
        "label": "DURATION",
        "pattern": [
            {"TEXT": {"REGEX": r"\d+[-]"}},
            {"LOWER": {"REGEX": r"days?|weeks?|months?|years?"}}
        ]
    },

    # thirty (30) days
    {
        "label": "DURATION",
        "pattern": [
            {"LOWER": {"REGEX": r"[a-z]+"}},
            {"TEXT": "("},
            {"LIKE_NUM": True},
            {"TEXT": ")"},
            {"LOWER": {"REGEX": r"days?|weeks?|months?|years?"}}
        ]
    }
]

patterns.extend(duration_patterns)


# ============================================================
# ADD PATTERNS
# ============================================================

ruler.add_patterns(patterns)

print("Legal regex patterns loaded successfully.")


# ============================================================
# YOUR EXISTING LEGAL LABELS
# ============================================================

legal_labels = {
    "ORG",
    "PERSON",
    "GPE",
    "LOC",
    "DATE",
    "MONEY",
    "LAW"
}


# ============================================================
# PROCESS YOUR EXISTING TEXT
# ============================================================

df = pd.DataFrame({
    "context_text": [text]
})

unique_texts = (
    df["context_text"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .tolist()
)

entity_map = {}

for i, doc in enumerate(
    nlp.pipe(unique_texts, batch_size=32)
):

    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        }
        for ent in doc.ents
        if (
            ent.label_ in legal_labels
            or ent.label_ in {
                "TERMINATION",
                "NOTICE_PERIOD",
                "EXACT_DATE",
                "DURATION",
                "EFFECTIVE_DATE",
                "AGREEMENT_DATE",
                "EXPIRATION",
                "RENEWAL",
                "CONFIDENTIALITY",
                "GOVERNING_LAW",
                "JURISDICTION",
                "INDEMNIFICATION",
                "LIABILITY",
                "WARRANTY",
                "ASSIGNMENT",
                "INTELLECTUAL_PROPERTY",
                "FORCE_MAJEURE",
                "DISPUTE_RESOLUTION",
                "PAYMENT",
                "NON_COMPETE"
            }
        )
    ]

    entity_map[unique_texts[i]] = entities


# ============================================================
# MAP BACK TO DATAFRAME
# ============================================================

df["legal_entities"] = (
    df["context_text"].map(entity_map)
)


# ============================================================
# GROUP LEGAL ENTITIES BY LABEL
# ============================================================

st.subheader("Extracted Legal Entities")

entities_df = pd.DataFrame(df["legal_entities"].iloc[0])

if not entities_df.empty:

    grouped_entities = (
    entities_df
    .groupby("label")["text"]
    .apply(lambda x: ", ".join(x))
    .reset_index()
)

grouped_entities.columns = ["Label", "Extracted Entities"]

selected_label = st.selectbox("Select a label to view extracted entities:", options=grouped_entities["Label"].unique(), key="label_dropdown") 
selected_entities = grouped_entities[grouped_entities["Label"] == selected_label]["Extracted Entities"].values[0]
st.write(f"Extracted entities for label '{selected_label}': {selected_entities}")




# ============================================================
# TOKENIZATION
# ============================================================


inputs = tokenizer(
    question,
    text,
    return_tensors="pt",
    truncation="only_second",
    max_length=384,
    return_offsets_mapping=True
)

offset_mapping = inputs.pop("offset_mapping")

# Identify which tokens belong to context
sequence_ids = inputs.sequence_ids(0)

inputs = {
    k: v.to(device)
    for k, v in inputs.items()
}


# ============================================================
# MODEL PREDICTION
# ============================================================

with torch.no_grad():
    outputs = model(**inputs)


MAX_ANSWER_LENGTH = 10

start_logits = outputs.start_logits[0]
end_logits = outputs.end_logits[0]
# ============================================================
# FIND BEST VALID ANSWER SPAN
# ============================================================

best_score = -float("inf")
best_start = None
best_end = None

for start in range(len(start_logits)):

    # Only context tokens
    if sequence_ids[start] != 1:
        continue

    for end in range(start, min(start + 30, len(end_logits))):

        if sequence_ids[end] != 1:
            continue

        score = (
            start_logits[start].item()
            + end_logits[end].item()
        )

        if score > best_score:
            best_score = score
            best_start = start
            best_end = end


# ============================================================
# EXTRACT ANSWER
# ============================================================

if best_start is not None:

    answer = tokenizer.decode(
        inputs["input_ids"][0][best_start:best_end + 1],
        skip_special_tokens=True
    )

else:
    answer = ""


# ============================================================
# OUTPUT
# ============================================================

st.write("Answer: " + answer)
st.write("Best score: " + str(best_score))
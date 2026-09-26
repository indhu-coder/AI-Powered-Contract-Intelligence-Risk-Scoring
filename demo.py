import streamlit as st
import json
import pandas as pd
import spacy
import fitz
import ocrmypdf
import tempfile
import os


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        pdf_path = temp_file.name

    ocr_output = pdf_path.replace(
        ".pdf",
        "_ocr.pdf"
    )

    try:

        # ----------------------------------------------------
        # STEP 1: Try PyMuPDF
        # ----------------------------------------------------

        doc = fitz.open(pdf_path)

        pages_text = []

        for page in doc:
            text = page.get_text("text")
            pages_text.append(text.strip())

        doc.close()

        extracted_text = "\n\n".join(pages_text)

        # ----------------------------------------------------
        # STEP 2: Check if OCR is needed
        # ----------------------------------------------------

        if len(extracted_text.strip()) > 100:
            return extracted_text

        # ----------------------------------------------------
        # STEP 3: OCR fallback
        # ----------------------------------------------------

        st.info(
            "Scanned PDF detected. Running OCR..."
        )

        ocrmypdf.ocr(
            pdf_path,
            ocr_output,
            deskew=True,
            force_ocr=True
        )

        # ----------------------------------------------------
        # STEP 4: Extract OCR text
        # ----------------------------------------------------

        doc = fitz.open(ocr_output)

        pages_text = []

        for page in doc:
            text = page.get_text("text")
            pages_text.append(text.strip())

        doc.close()

        return "\n\n".join(pages_text)

    finally:

        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        if os.path.exists(ocr_output):
            os.remove(ocr_output)


# ============================================================
# STREAMLIT FILE UPLOAD
# ============================================================


st.title("📄 Contract Intelligence AI")

uploaded_file = st.file_uploader(
    "Upload Contract JSON or PDF",
    type=["json", "pdf"]
)


if uploaded_file is not None:

    try:

        # ====================================================
        # OPTION 1: CUAD JSON
        # ====================================================

        if uploaded_file.type == "application/json":

            st.info("JSON file detected.")

            data = json.load(uploaded_file)

            rows = []

            for contract in data["data"]:

                contract_title = contract.get(
                    "title",
                    ""
                )

                contract_title = (
                    contract_title
                    .replace("-", "_")
                    .split("_")[-1]
                )

                for paragraph in contract.get(
                    "paragraphs",
                    []
                ):

                    context = paragraph.get(
                        "context",
                        ""
                    )

                    for qa in paragraph.get(
                        "qas",
                        []
                    ):

                        qa_id = qa.get(
                            "id",
                            ""
                        )

                        details = qa.get(
                            "details",
                            {}
                        )

                        field = qa_id.split(
                            "__"
                        )[-1]

                        answers = qa.get(
                            "answers",
                            []
                        )

                        # ------------------------------------
                        # Multiple answers
                        # ------------------------------------

                        if answers:

                            for answer in answers:

                                rows.append({

                                    "contract_title":
                                        contract_title,

                                    "qa_id":
                                        qa_id,

                                    "details":
                                        details,

                                    "field":
                                        field,

                                    "question":
                                        qa.get(
                                            "question",
                                            ""
                                        ),

                                    "context_text":
                                        context,

                                    "answer_text":
                                        answer.get(
                                            "text",
                                            ""
                                        ),

                                    "answer_start":
                                        answer.get(
                                            "answer_start",
                                            None
                                        ),

                                    "is_impossible":
                                        qa.get(
                                            "is_impossible",
                                            False
                                        )
                                })

                        # ------------------------------------
                        # No answer / impossible question
                        # ------------------------------------

                        else:

                            rows.append({

                                "contract_title":
                                    contract_title,

                                "qa_id":
                                    qa_id,

                                "details":
                                    details,

                                "field":
                                    field,

                                "question":
                                    qa.get(
                                        "question",
                                        ""
                                    ),

                                "context_text":
                                    context,

                                "answer_text":
                                    "",

                                "answer_start":
                                    None,

                                "is_impossible":
                                    qa.get(
                                        "is_impossible",
                                        True
                                    )
                            })

            df = pd.DataFrame(rows)

            st.success(
                "✅ JSON data loaded successfully."
            )


        # ====================================================
        # OPTION 2: PDF
        # ====================================================

        elif uploaded_file.type == "application/pdf":

            st.info("PDF file detected.")

            with st.spinner(
                "Extracting contract text..."
            ):

                text = extract_text_from_pdf(
                    uploaded_file
                )

            st.success(
                "✅ PDF processing completed."
            )

            st.subheader(
                "📄 Extracted Contract Text"
            )

            st.text_area(
                "Contract text",
                text,
                height=400
            )

            # Same context_text column used by NER

            df = pd.DataFrame({

                "contract_title": [
                    uploaded_file.name
                ],

                "field": [
                    "PDF Contract"
                ],

                "context_text": [
                    text
                ]

            })

        else:

            st.error(
                "Unsupported file type."
            )

            st.stop()


        # ====================================================
        # DATAFRAME PREVIEW
        # ====================================================

        st.subheader(
            "📊 DataFrame Preview"
        )

        st.write(
            f"Total rows: {len(df):,}"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )


        # ====================================================
        # NLP NER PIPELINE
        # ====================================================

        st.subheader(
            "🔎 Named Entity Recognition"
        )

        # ----------------------------------------------------
        # Load SpaCy
        # ----------------------------------------------------

        with st.spinner(
            "Loading SpaCy NER model..."
        ):

            nlp = spacy.load(
                "en_core_web_sm"
            )

        st.success(
            "SpaCy NER model loaded successfully."
        )


        # ----------------------------------------------------
        # Legal labels
        # ----------------------------------------------------

        legal_labels = {

            "ORG",
            "PERSON",
            "GPE",
            "LOC",
            "DATE",
            "MONEY",
            "LAW"

        }


        # ----------------------------------------------------
        # Process UNIQUE contexts only
        # ----------------------------------------------------

        unique_texts = (

            df["context_text"]

            .dropna()

            .astype(str)

            .drop_duplicates()

            .tolist()

        )

       


        # ----------------------------------------------------
        # NER processing
        # ----------------------------------------------------

        entity_map = {}

        progress_bar = st.progress(0)

        total_contexts = len(unique_texts)

        for i, doc in enumerate(
            nlp.pipe(
                unique_texts,
                batch_size=32
            )
        ):

            entities = [

                {
                    "text": ent.text,
                    "label": ent.label_
                }

                for ent in doc.ents

                if ent.label_ in legal_labels

            ]

            entity_map[
                unique_texts[i]
            ] = entities

            progress_bar.progress(
                (i + 1) / total_contexts
            )


        # ====================================================
        # MAP NER RESULTS BACK TO DATAFRAME
        # ====================================================

        df["legal_entities"] = (

            df["context_text"]

            .map(entity_map)

        )


        st.success(
            "✅ NER completed successfully."
        )


        # ====================================================
        # NER RESULTS
        # ====================================================

        st.subheader(
            "📊 DataFrame with NER Results"
        )

        st.write(
            f"Total rows: {len(df):,}"
        )

        st.dataframe(
            df[
                [
                    "contract_title",
                    "field",
                    "context_text",
                    "legal_entities"
                ]
            ].head(10),
            use_container_width=True
        )


        # ====================================================
        # ENTITY SUMMARY
        # ====================================================

        st.subheader(
            "🏷️ Extracted Legal Entities"
        )

        all_entities = []

        for entities in df[
            "legal_entities"
        ].dropna():

            all_entities.extend(
                entities
            )

        if all_entities:

            entity_df = pd.DataFrame(
                all_entities
            )

            st.dataframe(
                entity_df,
                use_container_width=True
            )

        else:

            st.info(
                "No legal entities were detected."
            )


    except Exception as e:

        st.error(
            f"Error processing file: {e}"
        )
# ContractIQ — AI-Powered Contract Intelligence and Risk Scoring

ContractIQ is an AI-assisted contract analysis project designed to help legal, compliance, and business teams review contracts more efficiently. It processes contract documents, answers questions about important clauses, extracts contract-related dates and terms, and supports risk-oriented review.

> **Important:** ContractIQ is a decision-support prototype, not a substitute for legal advice. Extracted answers and risk indicators must be reviewed by a qualified person before being used to make legal or business decisions.

## Project Goals

- Reduce the time required to find important information in long contracts.
- Answer clause-specific questions using a saved fine-tuned BERT question-answering model.
- Help identify dates, notice periods, termination terms, parties, jurisdictions, and other contract details when supported by the document.
- Support contract risk review through the project's risk-scoring and classification components.
- Provide an interactive interface for reviewing extracted answers and exporting results.

## Key Features

- **Document input:** Upload contract documents through the Streamlit interface. PDF and DOCX support is part of the application workflow.
- **Text extraction:** Extract text from documents before analysis. Scanned PDFs may require OCR; extraction quality depends on the document and OCR output.
- **Question-based extraction:** Ask predefined or custom questions about contract clauses and terms.
- **Long-document handling:** Tokenize document text with offsets and process long contexts in overlapping windows.
- **Clause-focused retrieval:** Retrieve relevant portions of a contract before running question-answering inference.
- **Answer-span decoding:** Select candidate answer spans and apply confidence/no-answer thresholds to reduce unsupported answers.
- **Contract risk components:** The repository includes risk-engine and clause-model components intended to support risk scoring/classification.
- **Results export:** The Streamlit workflow supports exporting analysis results to CSV.
- **Monitoring:** The inference workflow records useful diagnostics such as elapsed time, number of windows, model version/source, and status.

## Technology Stack

- Python
- Streamlit
- Hugging Face Transformers and a saved BERT question-answering model
- PyTorch
- Document text extraction for PDF/DOCX files
- OCR support for scanned documents, where configured
- Retrieval/vector-store components present in the repository
- Pandas for tabular results and CSV export
- Docker and Docker Compose files are included in the repository

The exact dependencies and versions should be taken from `requirements.txt` and the current project configuration.

## How the Inference Workflow Works

1. **Upload a contract** through the application.
2. **Extract text** from the uploaded document. Scanned PDFs may need OCR.
3. **Choose questions** from the available question set or enter custom questions.
4. **Retrieve relevant contract text** for the selected clause or question.
5. **Tokenize and window the text** so that long documents can be processed within the model's input limits. Character offsets help map predicted token spans back to the original text.
6. **Run question-answering inference** with the saved model.
7. **Decode candidate spans** and apply confidence and no-answer thresholds.
8. **Review the output** and export results if needed.

The application may use a baseline model if the configured fine-tuned model cannot be loaded. Check the logs and model-source/status fields to confirm which model was used for a particular run.

## Repository Components

The repository has included components such as:

- `app.py` — Streamlit user interface only for Question and anwering.
- `api.py` — API entry point
- `contract_risk_engine.py` — contract risk scoring/analysis logic
- `evaluate_model.py` — model evaluation utilities
- `train_clause_model.py` — clause-model training workflow
- `build_vector_index.py` — vector-index construction
- `vector_store.py` — vector-store integration
- `inference.py` - Application with risk scoring and Q&A
- `calibrate_model.py` — model calibration utilities
- `Dockerfile` and `docker-compose.yml` — container configuration
- `configs/`, `data/`, `models/`, `reports/`, and tests — configuration, data, model artifacts, reports, and test resources

File availability and exact responsibilities may differ by branch/version. Refer to the current repository for the authoritative structure.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/indhu-coder/AI-Powered-Contract-Intelligence-Risk-Scoring.git
cd AI-Powered-Contract-Intelligence-Risk-Scoring
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the project uses OCR for scanned PDFs, install and configure the OCR dependencies required by the current code and operating system. Consult the repository configuration for the supported OCR setup.

### 4. Configure the model path

The inference code must point to a locally available model directory containing the model and tokenizer files. One path used during development was:

```text
D:\ContractIntelligence\ContractIQ-main\ContractIQ-main\ml\models\final
```

This is a development-machine path, not a portable path for every installation. Update the model configuration or environment-specific setting to match your local checkout. Do not commit large model weights unless the repository's storage strategy explicitly supports them.

### 5. Run the Streamlit application

From the directory containing `app.py`, run:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

### Optional: Docker

The repository includes a `Dockerfile` and `docker-compose.yml`. Review those files and the required environment variables/model mounts before building or starting the container. Model weights and contract documents may need to be mounted or provisioned separately.

## Model and Evaluation Notes

The project uses a saved fine-tuned BERT question-answering model for extracting answers from contract text. The inference pipeline includes retrieval, windowing, span decoding, and no-answer threshold handling. This README does not claim that every contract field is extracted correctly in every document.

One recorded evaluation on a **100-question subset** reported:

| Metric | Result |
|---|---:|
| Overall Exact Match (EM) | 78.00% |
| Overall token-level F1 | 78.40% |
| Answerable questions | 30 |
| Answerable EM | 33.33% |
| Answerable F1 | 34.67% |
| Unanswerable questions | 70 |
| Correct empty predictions among unanswerable questions | 68 / 70 |
| Overall empty-prediction rate | 87.00% |

**Interpretation:** The overall scores are strongly influenced by the model correctly abstaining on many unanswerable questions. The lower answerable-question scores show that extracting the correct clause text remains a significant limitation. Evaluate answerable and unanswerable questions separately, and validate outputs against the original contract.

## Known Limitations

- Answer extraction may be incomplete or inaccurate, especially for answerable questions.
- Dates and durations can be ambiguous when a contract describes events, notice periods, renewal terms, or termination conditions in different sections.
- OCR errors in scanned PDFs can reduce downstream extraction quality.
- Retrieval may miss the relevant clause, and windowing can affect whether a complete answer span is available to the model.
- Confidence scores and no-answer thresholds are heuristics; they are not guarantees of correctness.
- Model loading may fall back to a baseline model if the fine-tuned model is unavailable. Confirm the model source/status in logs before interpreting results.
- Risk scores are indicators for review, not definitive legal conclusions.

## Responsible Use and Data Privacy

- Do not upload confidential contracts to an environment unless you are authorized to do so and understand where the files and outputs are processed or stored.
- Keep API keys, credentials, and private configuration out of source control.
- Avoid committing real contracts, personal information, generated predictions containing sensitive text, or model artifacts that should remain private.
- Review `.gitignore` and deployment settings before publishing or deploying the project.
- Always have a qualified reviewer verify extracted terms and risk flags before acting on them.

## Testing and Validation

Use the repository's available tests and evaluation scripts, for example:

```bash
pytest
python evaluate_model.py
```

These commands are examples; check each script's arguments and required data/model paths before running. Evaluation results are only comparable when the dataset split, question set, preprocessing, model version, and thresholds are consistent.

## Future Improvements

Potential next steps include improving answerable-question extraction, evaluating retrieval recall, refining no-answer calibration, adding targeted tests for date and duration normalization, and expanding evaluation across contract types. These are improvement opportunities, not claims that the work is already complete.

## Disclaimer

ContractIQ is an AI-assisted research and review tool. It can make mistakes, omit important clauses, or misinterpret context. It does not provide legal advice and should not be used as the sole basis for legal, compliance, financial, or business decisions.

## Repository

GitHub: https://github.com/indhu-coder/AI-Powered-Contract-Intelligence-Risk-Scoring

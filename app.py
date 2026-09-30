"""ContractIQ - AI-Powered Contract Intelligence & Risk Scoring Web Application.

Run using:
    streamlit run app.py
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="ContractIQ | AI Contract Intelligence & Risk Scoring",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

# Custom styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .risk-card {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 6px solid #ccc;
    }
    .risk-high {
        background-color: #FEF2F2;
        border-left-color: #EF4444;
    }
    .risk-medium {
        background-color: #FFFBEB;
        border-left-color: #F59E0B;
    }
    .risk-low {
        background-color: #F0FDF4;
        border-left-color: #10B981;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-high { background-color: #FEE2E2; color: #991B1B; }
    .badge-medium { background-color: #FEF3C7; color: #92400E; }
    .badge-low { background-color: #DCFCE7; color: #166534; }
    .badge-info { background-color: #E0E7FF; color: #3730A3; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_engine():
    from contract_risk_engine import ContractIntelligenceEngine
    return ContractIntelligenceEngine()


@st.cache_data
def load_cuad_dataset():
    json_path = BASE_DIR / "cuad-main" / "data" / "CUADv1.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for contract in data.get("data", []):
        c_title = contract.get("title", "").replace("-", "_").split("_")[-1]
        for para in contract.get("paragraphs", []):
            ctx = para.get("context", "")
            for qa in para.get("qas", []):
                q_id = qa.get("id", "")
                field = q_id.split("__")[-1]
                answers = qa.get("answers", [])
                if answers:
                    for a in answers:
                        rows.append({
                            "contract_title": c_title,
                            "field": field,
                            "question": qa.get("question", ""),
                            "context_text": ctx,
                            "answer_text": a.get("text", ""),
                            "is_impossible": False,
                        })
                else:
                    rows.append({
                        "contract_title": c_title,
                        "field": field,
                        "question": qa.get("question", ""),
                        "context_text": ctx,
                        "answer_text": "",
                        "is_impossible": True,
                    })
    return pd.DataFrame(rows)


def extract_text_from_upload(uploaded_file) -> str:
    """Extract text from uploaded PDF, TXT or DOCX."""
    fname = uploaded_file.name.lower()
    content = uploaded_file.read()

    if fname.endswith(".pdf"):
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        pages = [page.get_text("text").strip() for page in doc]
        doc.close()
        text = "\n\n".join(pages)
        if len(text.strip()) < 20:
            return "Scanned PDF uploaded. No direct text layer found."
        return text

    elif fname.endswith(".docx"):
        import docx
        import io
        d = docx.Document(io.BytesIO(content))
        parts = [p.text for p in d.paragraphs if p.text.strip()]
        return "\n\n".join(parts)

    else:
        # Plain text
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1", errors="ignore")


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/contract.png", width=64)
st.sidebar.title("ContractIQ")
st.sidebar.markdown("**AI Legal Intelligence & Risk Scoring**")
menu = st.sidebar.radio(
    "Navigation",
    [
        "📑 Contract Analysis & Risk Scoring",
        "📊 CUAD Dataset & EDA",
        "📈 Model Training & Evaluation (Week 2)",
        "🔍 Semantic Search & Embeddings",
        "⚙️ Pipeline Execution & Logs",
    ],
)

engine = get_engine()

# ==============================================================================
# TAB 1: CONTRACT ANALYSIS & RISK SCORING
# ==============================================================================
if menu == "📑 Contract Analysis & Risk Scoring":
    st.markdown('<div class="main-header">Contract Intelligence & Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated legal clause extraction, entity recognition, and risk detection.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        source_mode = st.radio(
            "Select Contract Source:",
            ["Use Sample PDF Contract (sample_contract.pdf)", "Use Scanned/OCR PDF (ocr_output.pdf)", "Upload Custom Contract (PDF/DOCX/TXT)"],
        )

    contract_text = ""
    contract_title = "Contract"

    if source_mode == "Use Sample PDF Contract (sample_contract.pdf)":
        sample_path = BASE_DIR / "sample_contract.pdf"
        if sample_path.exists():
            import fitz
            doc = fitz.open(str(sample_path))
            contract_text = "\n\n".join(page.get_text("text").strip() for page in doc)
            doc.close()
            contract_title = "Consulting Agreement (sample_contract.pdf)"
            st.success(f"Loaded `{sample_path.name}` ({len(contract_text):,} characters)")
        else:
            st.error("sample_contract.pdf not found.")

    elif source_mode == "Use Scanned/OCR PDF (ocr_output.pdf)":
        ocr_path = BASE_DIR / "ocr_output.pdf"
        if ocr_path.exists():
            import fitz
            doc = fitz.open(str(ocr_path))
            contract_text = "\n\n".join(page.get_text("text").strip() for page in doc)
            doc.close()
            contract_title = "Commercial Agreement (ocr_output.pdf)"
            st.success(f"Loaded `{ocr_path.name}` ({len(contract_text):,} characters)")
        else:
            st.error("ocr_output.pdf not found.")

    else:
        uploaded_file = st.file_uploader("Upload contract file", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            contract_text = extract_text_from_upload(uploaded_file)
            contract_title = uploaded_file.name
            st.success(f"Loaded `{uploaded_file.name}` ({len(contract_text):,} characters)")

    if contract_text and len(contract_text.strip()) > 30:
        if st.button("🚀 Analyze Contract & Calculate Risk Score", type="primary", use_container_width=True):
            with st.spinner("Analyzing clauses, extracting legal entities, and evaluating risk heuristics..."):
                res = engine.analyze_contract(contract_text, title=contract_title)

            st.markdown("---")

            # Risk Summary Cards
            c_score, c_level, c_chars, c_words = st.columns(4)
            c_score.metric("Overall Risk Score", f"{res.overall_risk_score} / 100")
            c_level.metric("Risk Level", res.overall_risk_level)
            c_chars.metric("Character Count", f"{res.character_count:,}")
            c_words.metric("Word Count", f"{res.word_count:,}")

            # Risk Banner
            if res.overall_risk_level == "HIGH":
                st.error(f"⚠️ **HIGH RISK ({res.overall_risk_score}/100)**: Multiple high-exposure clauses detected. Thorough legal counsel review is strongly advised before signing.")
            elif res.overall_risk_level == "MEDIUM":
                st.warning(f"⚡ **MEDIUM RISK ({res.overall_risk_score}/100)**: Moderate risk detected. Certain terms require renegotiation or clarification.")
            else:
                st.success(f"✅ **LOW RISK ({res.overall_risk_score}/100)**: Contract adheres to standard protective legal terms.")

            # Section: Key Dates & Durations
            st.subheader("📅 Critical Dates & Durations")
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.write("**Identified Dates:**")
                for k, v in res.extracted_dates.items():
                    val = v if v else "Not specified"
                    st.write(f"- **{k}:** `{val}`")
            with d_col2:
                st.write("**Durations & Notice Periods:**")
                for k, v in res.durations.items():
                    val = v if v else "Not specified"
                    st.write(f"- **{k}:** `{val}`")

            # Section: Risk Findings
            st.subheader("🚩 Risk Findings & Red Flags")
            if res.risk_findings:
                for rf in res.risk_findings:
                    cls_name = "risk-high" if rf.severity == "HIGH" else ("risk-medium" if rf.severity == "MEDIUM" else "risk-low")
                    badge_cls = "badge-high" if rf.severity == "HIGH" else ("badge-medium" if rf.severity == "MEDIUM" else "badge-low")
                    st.markdown(
                        f"""
                        <div class="risk-card {cls_name}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <h4 style="margin:0; color:#111827;">{rf.title}</h4>
                                <span class="metric-badge {badge_cls}">{rf.severity} (+{rf.weight} pts)</span>
                            </div>
                            <p style="margin:0.5rem 0; color:#374151;">{rf.description}</p>
                            <p style="margin:0; font-size:0.9rem; color:#1E40AF;"><strong>Recommendation:</strong> {rf.recommendation}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No anomalous risk flags identified.")

            # Section: Detected Clauses
            st.subheader("📑 Detected Contract Clauses")
            clause_rows = []
            for c in res.clauses:
                clause_rows.append({
                    "Clause Name": c.display_name,
                    "Detected": "✅ Found" if c.found else "❌ Not Found",
                    "Confidence": f"{c.confidence * 100:.0f}%",
                    "Risk Level": c.risk_level,
                    "Excerpt": c.excerpt,
                })
            st.dataframe(pd.DataFrame(clause_rows), use_container_width=True)

            # Section: Legal Entities (NER)
            st.subheader("🏷️ Legal Entities (Named Entity Recognition)")
            e_tabs = st.tabs(["Parties (ORG/PERSON)", "Locations (GPE/LOC)", "Dates", "Monetary Amounts", "Governing Laws"])
            with e_tabs[0]:
                st.write(res.entities.get("parties", []))
            with e_tabs[1]:
                st.write(res.entities.get("locations", []))
            with e_tabs[2]:
                st.write(res.entities.get("dates", []))
            with e_tabs[3]:
                st.write(res.entities.get("money", []))
            with e_tabs[4]:
                st.write(res.entities.get("laws", []))

            # Full Document Text Preview
            with st.expander("📄 View Extracted Document Text"):
                st.text_area("Contract Text", contract_text, height=350)

# ==============================================================================
# TAB 2: CUAD DATASET EXPLORER & EDA
# ==============================================================================
elif menu == "📊 CUAD Dataset & EDA":
    st.markdown('<div class="main-header">CUAD Legal Dataset & Exploratory Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspection of 28,031 Contract QA pairs across 510 contracts (Henriksson et al., CUAD v1).</div>', unsafe_allow_html=True)

    with st.spinner("Loading CUAD dataset..."):
        df = load_cuad_dataset()

    if df is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total QA Instances", f"{len(df):,}")
        m2.metric("Unique Contracts", f"{df['contract_title'].nunique():,}")
        m3.metric("Legal Fields", f"{df['field'].nunique():,}")
        positive_count = (~df["is_impossible"]).sum()
        m4.metric("Positive Clause Spans", f"{positive_count:,}")

        st.subheader("Field Distribution")
        field_counts = df["field"].value_counts().reset_index()
        field_counts.columns = ["Clause / Field", "Count"]
        st.bar_chart(field_counts.set_index("Clause / Field").head(15))

        st.subheader("Browse Dataset by Field")
        selected_field = st.selectbox("Select Clause Type:", sorted(df["field"].unique()))
        filtered = df[df["field"] == selected_field]

        st.write(f"Showing **{len(filtered)}** QA instances for `{selected_field}`:")
        st.dataframe(
            filtered[["contract_title", "question", "answer_text", "is_impossible"]].head(50),
            use_container_width=True,
        )
    else:
        st.error("CUADv1.json not found in cuad-main/data/CUADv1.json")

# ==============================================================================
# TAB: MODEL TRAINING & EVALUATION (WEEK 2)
# ==============================================================================
elif menu == "📈 Model Training & Evaluation (Week 2)":
    st.markdown('<div class="main-header">Model Training & Evaluation (Week 2)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fine-tuning transformers on CUAD legal clauses, precision/recall metrics, and threshold calibration.</div>', unsafe_allow_html=True)

    t_rep_path = BASE_DIR / "reports" / "week2_training_report.json"
    e_rep_path = BASE_DIR / "reports" / "week2_evaluation_report.json"
    c_rep_path = BASE_DIR / "reports" / "week2_calibration_report.json"

    m_tab1, m_tab2, m_tab3 = st.tabs(["📊 Evaluation Metrics (F1 / Precision / Recall)", "🎯 Threshold Calibration & Heuristics", "🏋️ Fine-Tuning Checkpoints & Runner"])

    with m_tab1:
        st.subheader("Model Performance on Legal Clause Extraction")
        if e_rep_path.exists():
            with open(e_rep_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            ov = eval_data.get("overall", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Token F1 Score", f"{ov.get('f1', 0) * 100:.1f}%")
            c2.metric("Token Precision", f"{ov.get('precision', 0) * 100:.1f}%")
            c3.metric("Token Recall", f"{ov.get('recall', 0) * 100:.1f}%")
            c4.metric("Exact Match (EM)", f"{ov.get('exact_match', 0) * 100:.1f}%")

            st.write(f"Evaluated across **{ov.get('evaluated_spans', 0)}** contract clauses using model: `{eval_data.get('model_path', 'Baseline')}`")

            # Per clause table
            per_c = eval_data.get("per_clause", {})
            if per_c:
                st.subheader("Per-Clause Category Metrics")
                clause_rows = []
                for k, v in per_c.items():
                    clause_rows.append({
                        "Clause Category": k,
                        "F1 Score (%)": round(v["f1"] * 100, 1),
                        "Precision (%)": round(v["precision"] * 100, 1),
                        "Recall (%)": round(v["recall"] * 100, 1),
                        "Exact Match (%)": round(v["exact_match"] * 100, 1),
                        "Evaluated Count": v["count"],
                    })
                cdf = pd.DataFrame(clause_rows)
                st.dataframe(cdf, use_container_width=True)
                st.bar_chart(cdf.set_index("Clause Category")[["F1 Score (%)", "Precision (%)", "Recall (%)"]].head(10))
        else:
            st.info("Run `python evaluate_model.py` to generate the official evaluation report.")
            if st.button("Run Model Evaluation Now", type="primary"):
                with st.spinner("Evaluating model precision/recall..."):
                    import subprocess
                    subprocess.run([sys.executable, "evaluate_model.py", "--quick"], cwd=str(BASE_DIR))
                st.rerun()

    with m_tab2:
        st.subheader("Threshold Calibration (Macro F1 Optimization)")
        if c_rep_path.exists():
            with open(c_rep_path, "r", encoding="utf-8") as f:
                cal_data = json.load(f)

            best = cal_data.get("optimal_thresholds", {})
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Optimal Min Confidence", best.get("min_confidence", 0.35))
            b2.metric("Optimal No-Answer Delta", best.get("no_answer_delta", 0.05))
            b3.metric("Calibrated Precision", f"{best.get('precision', 0) * 100:.1f}%")
            b4.metric("Calibrated Recall", f"{best.get('recall', 0) * 100:.1f}%")

            st.write("### Post-Processing Legal Heuristics Applied")
            st.json(cal_data.get("post_processing_heuristics", {}))
        else:
            st.info("Run `python calibrate_model.py` to generate optimal decision thresholds.")
            if st.button("Run Threshold Calibration Now", type="primary"):
                with st.spinner("Calibrating thresholds..."):
                    import subprocess
                    subprocess.run([sys.executable, "calibrate_model.py"], cwd=str(BASE_DIR))
                st.rerun()

    with m_tab3:
        st.subheader("Fine-Tuned Checkpoint Information")
        chk_dir = BASE_DIR / "models" / "fine_tuned_clause_model"
        if chk_dir.exists():
            st.success(f"✅ Active Checkpoint: `{chk_dir}`")
            files = [f.name for f in chk_dir.iterdir()]
            st.write(f"Checkpoint Files: `{', '.join(files)}`")
        else:
            st.warning("No local checkpoint found yet under `models/fine_tuned_clause_model`.")

        if t_rep_path.exists():
            with open(t_rep_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
            st.write("### Latest Training Run")
            st.json(t_data)

# ==============================================================================
# TAB 3: SEMANTIC SEARCH & EMBEDDINGS
# ==============================================================================
elif menu == "🔍 Semantic Search & Embeddings":
    st.markdown('<div class="main-header">Semantic Contract Search & Embeddings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Query contracts using SentenceTransformer (`all-MiniLM-L6-v2`) dense vector embeddings.</div>', unsafe_allow_html=True)

    query = st.text_input("Enter legal query or clause requirement:", "limitation of liability and indemnification cap")
    top_k_select = st.slider("Number of results to retrieve:", min_value=1, max_value=20, value=5)

    v_dir = BASE_DIR / "data" / "vector_store"
    from vector_store import VectorStore

    store = VectorStore()
    has_index = store.load(v_dir)

    c_s1, c_s2 = st.columns([1, 1])
    with c_s1:
        if has_index:
            st.success(f"⚡ FAISS Vector Store active: **{store.count():,}** passages indexed.")
        else:
            st.warning("FAISS index not found. Click below to build.")
            if st.button("Build Vector Index Now"):
                with st.spinner("Indexing contracts into FAISS..."):
                    import subprocess
                    subprocess.run([sys.executable, "build_vector_index.py", "--limit", "30"], cwd=str(BASE_DIR))
                st.rerun()

    if st.button("Search Contracts with FAISS", type="primary"):
        with st.spinner("Searching FAISS vector database..."):
            if has_index and store.count() > 0:
                results = store.search(query, top_k=top_k_select, min_score=0.1)
                st.subheader(f"Top {len(results)} Relevant Contract Passages")
                for r in results:
                    st.markdown(
                        f"""
                        <div class="risk-card risk-low">
                            <div style="display:flex; justify-content:space-between;">
                                <h4 style="margin:0; color:#1E3A8A;">{r.get('contract_title', 'Contract')}</h4>
                                <span class="metric-badge badge-info">Cosine Similarity: {r['similarity_score']:.4f}</span>
                            </div>
                            <p style="margin:0.5rem 0; font-size:0.92rem; color:#374151;">{r['text']}</p>
                            <span style="font-size:0.8rem; color:#6B7280;">Passage ID: {r.get('id', 'N/A')} | Char range: [{r.get('char_start')}:{r.get('char_end')}]</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.error("Vector index is empty. Please build it first.")

# ==============================================================================
# TAB 4: PIPELINE EXECUTION & LOGS
# ==============================================================================
elif menu == "⚙️ Pipeline Execution & Logs":
    st.markdown('<div class="main-header">Pipeline Execution & Terminal Runner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Execute the underlying Python pipeline scripts and view outputs in real-time.</div>', unsafe_allow_html=True)

    script_choice = st.selectbox(
        "Choose script to run:",
        [
            "ocrpipeline.py (PDF Text Extraction & PyMuPDF)",
            "main.py (CUAD Dataset Loading & Date/Duration Analysis)",
            "transformer_model.py (SentenceTransformer Embeddings)",
        ],
    )

    if st.button("▶️ Run Selected Script", type="primary"):
        import subprocess

        cmd = ""
        if "ocrpipeline" in script_choice:
            cmd = [sys.executable, "ocrpipeline.py"]
        elif "main.py" in script_choice:
            cmd = [sys.executable, "main.py"]
        elif "transformer_model" in script_choice:
            cmd = [sys.executable, "transformer_model.py"]

        with st.spinner(f"Running {cmd[1]}..."):
            proc = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)

        if proc.returncode == 0:
            st.success(f"`{cmd[1]}` executed successfully with exit code 0!")
        else:
            st.error(f"`{cmd[1]}` exited with code {proc.returncode}")

        st.subheader("Console Output (Stdout):")
        st.code(proc.stdout if proc.stdout else "(No output)")

        if proc.stderr:
            st.subheader("Console Errors (Stderr):")
            st.code(proc.stderr)

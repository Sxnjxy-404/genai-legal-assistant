# ⚖️ GenAI Legal Assistant for SMEs

This project is a Contract Analysis & Risk Assessment Bot built for the GUVI–HCL Hackathon.

It helps small and medium business owners understand complex contracts, identify risky clauses, and receive safer alternatives in plain language.

---

## 🚀 Features

- Upload contracts (PDF, DOCX, TXT)
- Clause & sub-clause extraction
- Contract type classification (Employment, Vendor, Lease, Partnership, Service)
- Named Entity Recognition (Parties, Dates, Amounts, Jurisdiction)
- Clause-level risk scoring (High / Medium / Low)
- Obligation vs Right vs Prohibition detection
- Unfavorable clause highlighting
- Standardized SME-friendly contract templates
- Audit trail generation (JSON log)
- PDF export of summary report
- Multilingual UI support (English & Hindi)
- Real LLM (GPT-4) support for:
  - Clause explanation
  - Safer alternative suggestions
  - Summary generation  
  (disabled in demo due to billing)

---

## 🧠 Tech Stack

- Python
- Streamlit (UI)
- spaCy (NLP)
- OpenAI GPT-4 (for reasoning)
- pdfplumber, python-docx
- ReportLab (PDF export)

---

## ▶ How to Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py

## 🔐 LLM Usage

The app supports real LLM integration.  
To enable it:

1. Set environment variable:

```bash
setx OPENAI_API_KEY "your_api_key"

2.In app.py, change:

USE_LLM = True


## 📂 Output Files

audit_log.json → stores audit trail

contract_report.pdf → exported summary
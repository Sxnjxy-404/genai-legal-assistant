import streamlit as st
import pdfplumber, docx, re, json, os
import spacy
import stanza
from langdetect import detect
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ================= CONFIG =================
USE_LLM = False  # turn True when billing enabled

if USE_LLM:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------- Load NLP Models --------
nlp_en = spacy.load("en_core_web_sm")

# Download & load Hindi model (first run only)
stanza.download("hi")
nlp_hi = stanza.Pipeline("hi")

st.set_page_config(page_title="GenAI Legal Assistant", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.risk-high {
    background-color:#ffcccc;
    color:#000000;
    padding:15px;
    border-radius:10px;
    font-size:16px;
}
.risk-medium {
    background-color:#fff3cd;
    color:#000000;
    padding:15px;
    border-radius:10px;
    font-size:16px;
}
.risk-low {
    background-color:#d4edda;
    color:#000000;
    padding:15px;
    border-radius:10px;
    font-size:16px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ GenAI Legal Assistant for SMEs")
st.write("Understand contracts, detect risks, and get safer alternatives.")

# ---------------- CONTRACT TEMPLATES ----------------
TEMPLATES = {
    "Employment": """EMPLOYMENT AGREEMENT TEMPLATE
1. Employee shall work as per company policies.
2. Salary shall be paid monthly.
3. Either party may terminate with 30 days notice.
4. Confidentiality must be maintained.
""",
    "Vendor": """VENDOR AGREEMENT TEMPLATE
1. Vendor shall supply goods as agreed.
2. Payment shall be made within 30 days.
3. Liability is limited to contract value.
4. Disputes resolved by mutual discussion.
""",
    "Lease": """LEASE AGREEMENT TEMPLATE
1. Tenant shall pay rent monthly.
2. Lease term is 11 months.
3. Property must be used legally.
4. Termination requires 30 days notice.
""",
    "Partnership": """PARTNERSHIP DEED TEMPLATE
1. Partners shall share profits equally.
2. All partners must act in good faith.
3. Disputes resolved by arbitration.
4. Partnership may dissolve mutually.
""",
    "Service": """SERVICE CONTRACT TEMPLATE
1. Service provider shall deliver agreed services.
2. Client shall pay as per invoice.
3. Confidentiality must be maintained.
4. Termination with prior notice.
"""
}

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Contract", type=["pdf", "docx", "txt"])

# ---------------- TEXT EXTRACTION ----------------
def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages:
                if p.extract_text():
                    text += p.extract_text() + "\n"
    elif "word" in file.type:
        d = docx.Document(file)
        for para in d.paragraphs:
            text += para.text + "\n"
    else:
        text = file.read().decode("utf-8")
    return text

# ---------------- LANGUAGE + CLAUSE SPLIT ----------------
def get_clauses(text):
    try:
        lang = detect(text)
    except:
        lang = "en"

    if lang == "hi":
        doc = nlp_hi(text)
        clauses = [sent.text for sent in doc.sentences]
        return clauses, "Hindi"
    else:
        doc = nlp_en(text)
        clauses = [sent.text for sent in doc.sents]
        return clauses, "English"

# ---------------- CONTRACT TYPE CLASSIFIER ----------------
def classify_contract(text):
    t = text.lower()
    if "employee" in t or "salary" in t:
        return "Employment"
    if "vendor" in t or "supply" in t:
        return "Vendor"
    if "rent" in t or "lease" in t:
        return "Lease"
    if "partner" in t or "partnership" in t:
        return "Partnership"
    if "service" in t:
        return "Service"
    return "General"

# ---------------- NER (English only) ----------------
def extract_entities(text, lang):
    if lang == "Hindi":
        doc = nlp_hi(text)
        ents = {"PERSON": [], "ORG": [], "DATE": [], "MONEY": [], "LOCATION": []}
        for sent in doc.sentences:
            for ent in sent.ents:
                if ent.type in ents:
                    ents[ent.type].append(ent.text)
        return ents
    else:
        doc = nlp_en(text)
        ents = {"PARTY": [], "DATE": [], "MONEY": [], "GPE": []}
        for e in doc.ents:
            if e.label_ in ["ORG", "PERSON"]:
                ents["PARTY"].append(e.text)
            elif e.label_ == "DATE":
                ents["DATE"].append(e.text)
            elif e.label_ == "MONEY":
                ents["MONEY"].append(e.text)
            elif e.label_ == "GPE":
                ents["GPE"].append(e.text)
        return ents

# ---------------- RISK KEYWORDS ----------------
HIGH_EN = ["indemnify", "penalty", "terminate", "non-compete", "liability", "damages"]
MEDIUM_EN = ["arbitration", "jurisdiction", "auto-renewal", "lock-in", "governing law"]
LOW_EN = ["payment", "notice", "confidentiality"]

HIGH_HI = ["क्षतिपूर्ति", "दंड", "जुर्माना", "समाप्त", "उत्तरदायित्व"]
MEDIUM_HI = ["मध्यस्थता", "क्षेत्राधिकार", "नवीकरण", "कानून"]
LOW_HI = ["भुगतान", "सूचना", "गोपनीयता"]

# ---------------- RISK ENGINE ----------------
def risk_score(clause, lang):
    t = clause.lower()
    if lang == "Hindi":
        for w in HIGH_HI:
            if w in t: return "High"
        for w in MEDIUM_HI:
            if w in t: return "Medium"
        for w in LOW_HI:
            if w in t: return "Low"
    else:
        for w in HIGH_EN:
            if w in t: return "High"
        for w in MEDIUM_EN:
            if w in t: return "Medium"
        for w in LOW_EN:
            if w in t: return "Low"
    return "Low"

def obligation_type(clause):
    c = clause.lower()
    if "shall not" in c or "नहीं करेगा" in c: return "Prohibition"
    if "shall" in c or "must" in c or "करेगा" in c: return "Obligation"
    if "may" in c or "सकता है" in c: return "Right"
    return "Neutral"

# ---------------- PDF EXPORT ----------------
def export_pdf(report):
    file = "contract_report.pdf"
    c = canvas.Canvas(file, pagesize=A4)
    t = c.beginText(40, 800)
    for line in report.split("\n"):
        t.textLine(line)
    c.drawText(t)
    c.save()
    return file

# ---------------- MAIN ----------------
if uploaded_file:
    text = extract_text(uploaded_file)

    clauses, lang = get_clauses(text)
    st.subheader(f"🌐 Detected Language: {lang}")

    contract_type = classify_contract(text)
    st.subheader(f"📂 Detected Contract Type: {contract_type}")

    ents = extract_entities(text, lang)
    st.subheader("📌 Extracted Entities")
    st.json(ents)

    audit = {"time": str(datetime.now()), "language": lang, "contract_type": contract_type, "clauses": []}

    high, med = 0, 0
    st.subheader("📑 Clause Analysis")

    for i, cl in enumerate(clauses):
        r = risk_score(cl, lang)
        o = obligation_type(cl)

        css = "risk-low"
        if r == "High":
            css = "risk-high"
            high += 1
        elif r == "Medium":
            css = "risk-medium"
            med += 1

        st.markdown(
            f"<div class='{css}'><b>Clause {i+1} - {r} Risk</b><br><br>{cl}<br><br><b>Type:</b> {o}</div>",
            unsafe_allow_html=True
        )

        audit["clauses"].append({"clause": cl, "risk": r, "type": o})
        st.divider()

    st.subheader("📊 Contract Risk")
    if high > 2:
        st.error("Overall Risk: HIGH")
    elif med > 2:
        st.warning("Overall Risk: MEDIUM")
    else:
        st.success("Overall Risk: LOW")

    summary = "Summary not generated (AI disabled)."
    st.subheader("📝 Summary Report")
    st.write(summary)

    with open("audit_log.json","w") as f:
        json.dump(audit,f,indent=2)

    if st.button("Export PDF"):
        pdf = export_pdf(summary)
        st.success("PDF created: " + pdf)

    st.subheader("📄 SME-Friendly Contract Templates")
    temp_choice = st.selectbox("Choose Template", list(TEMPLATES.keys()))
    st.text_area("Template Preview", TEMPLATES[temp_choice], height=200)

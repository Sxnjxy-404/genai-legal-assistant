import streamlit as st
import pdfplumber, docx, re, json, os
import spacy
import stanza
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ================= CONFIG =================
USE_LLM = False  # turn True when billing enabled

if USE_LLM:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- LOAD NLP MODELS ----------------
@st.cache_resource
def load_models():
    nlp_en = spacy.load("en_core_web_sm")
    stanza.download("hi")
    nlp_hi = stanza.Pipeline("hi", processors="tokenize,ner")
    return nlp_en, nlp_hi

nlp_en, nlp_hi = load_models()

st.set_page_config(page_title="GenAI Legal Assistant", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.risk-high {background-color:#ffcccc;color:black;padding:15px;border-radius:10px;}
.risk-medium {background-color:#fff3cd;color:black;padding:15px;border-radius:10px;}
.risk-low {background-color:#d4edda;color:black;padding:15px;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ GenAI Legal Assistant for SMEs")
st.write("Understand contracts, detect risks, and get safer alternatives.")

# ---------------- CONTRACT TEMPLATES ----------------
TEMPLATES = {
    "Employment": "Employee shall work as per company policies.\nSalary paid monthly.\n30 days termination notice.\nConfidentiality required.",
    "Vendor": "Vendor shall supply goods.\nPayment within 30 days.\nLiability limited.\nDisputes by discussion.",
    "Lease": "Tenant pays rent monthly.\n11 month lease.\nLegal use only.\n30 days termination.",
    "Partnership": "Profits shared equally.\nPartners act in good faith.\nArbitration for disputes.",
    "Service": "Service provider delivers services.\nClient pays invoice.\nConfidentiality applies."
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

# ---------------- LANGUAGE DETECTION ----------------
def detect_language(text):
    if re.search(r"[अ-ह]", text):
        return "Hindi"
    return "English"

# ---------------- CLAUSE SPLIT ----------------
def get_clauses(text, lang):
    if lang == "Hindi":
        doc = nlp_hi(text)
        return [s.text for s in doc.sentences]
    else:
        doc = nlp_en(text)
        return [s.text for s in doc.sents]

# ---------------- CONTRACT TYPE ----------------
def classify_contract(text):
    t = text.lower()
    if "employee" in t or "salary" in t: return "Employment"
    if "vendor" in t or "supply" in t: return "Vendor"
    if "rent" in t or "lease" in t: return "Lease"
    if "partner" in t: return "Partnership"
    if "service" in t: return "Service"
    return "General"

# ---------------- ENTITY EXTRACTION ----------------
def extract_entities(text, lang):
    ents = {"PERSON": [], "ORG": [], "DATE": [], "MONEY": [], "LOCATION": []}

    if lang == "Hindi":
        # -------- REGEX-BASED HINDI ENTITY EXTRACTION --------
        # PERSON (श्री <नाम>)
        persons = re.findall(r"श्री\s+[अ-ह]+\s*[अ-ह]*", text)
        ents["PERSON"].extend(persons)

        # ORG (कंपनी नाम / लिमिटेड)
        orgs = re.findall(r"[A-Za-zअ-ह]+\s+(प्राइवेट लिमिटेड|लिमिटेड|कंपनी)", text)
        ents["ORG"].extend(orgs)

        # DATE (1 जनवरी 2025)
        dates = re.findall(r"\d{1,2}\s+[अ-ह]+\s+\d{4}", text)
        ents["DATE"].extend(dates)

        # MONEY (₹50000 or 50000 रुपये)
        money = re.findall(r"₹\s?\d+|\d+\s?रुपये", text)
        ents["MONEY"].extend(money)

    else:
        doc = nlp_en(text)
        for e in doc.ents:
            if e.label_ == "PERSON":
                ents["PERSON"].append(e.text)
            elif e.label_ == "ORG":
                ents["ORG"].append(e.text)
            elif e.label_ == "DATE":
                ents["DATE"].append(e.text)
            elif e.label_ == "MONEY":
                ents["MONEY"].append(e.text)
            elif e.label_ in ["GPE", "LOC"]:
                ents["LOCATION"].append(e.text)

    return ents


# ---------------- RISK KEYWORDS ----------------
HIGH_EN = ["indemnify", "penalty", "terminate", "liability", "damages"]
MEDIUM_EN = ["arbitration", "jurisdiction", "lock-in"]
LOW_EN = ["payment", "notice", "confidentiality"]

HIGH_HI = ["क्षतिपूर्ति", "दंड", "समाप्त", "उत्तरदायित्व"]
MEDIUM_HI = ["मध्यस्थता", "क्षेत्राधिकार"]
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

    lang = detect_language(text)
    st.subheader(f"🌐 Detected Language: {lang}")

    clauses = get_clauses(text, lang)

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

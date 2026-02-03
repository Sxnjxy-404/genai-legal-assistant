import streamlit as st
import pdfplumber, docx, re, json, os
import spacy
import stanza
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ================= CONFIG =================
USE_LLM = False   # turn True when you add API key

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
.ambiguous {border-left:6px solid red;}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ GenAI Legal Assistant for SMEs")

# ---------------- CONTRACT TEMPLATES ----------------
TEMPLATES = {
    "Employment": [
        "Employee shall work as per company policies.",
        "Salary shall be paid monthly.",
        "Either party may terminate with 30 days notice.",
        "Confidentiality must be maintained."
    ],
    "Service": [
        "Service provider shall deliver services.",
        "Client shall pay as per invoice."
    ]
}

uploaded_file = st.file_uploader("Upload Contract", type=["pdf", "docx", "txt"])

# ---------------- TEXT EXTRACTION ----------------
def extract_text(file):
    if file.type == "application/pdf":
        text = ""
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages:
                if p.extract_text():
                    text += p.extract_text() + "\n"
        return text
    elif "word" in file.type:
        d = docx.Document(file)
        return "\n".join(p.text for p in d.paragraphs)
    else:
        return file.read().decode("utf-8")

# ---------------- LANGUAGE DETECTION ----------------
def detect_language(text):
    if re.search(r"[अ-ह]", text):
        return "Hindi"
    return "English"

# ---------------- CLAUSE SPLIT ----------------
def get_clauses(text, lang):
    if lang == "Hindi":
        doc = nlp_hi(text)
        clauses = [s.text.strip() for s in doc.sentences if s.text.strip()]
        if not clauses:
            clauses = re.split(r"[।\n]", text)
        return clauses
    else:
        doc = nlp_en(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]

# ---------------- CONTRACT TYPE ----------------
def classify_contract(text):
    t = text.lower()
    if "employee" in t: return "Employment"
    if "service" in t: return "Service"
    return "General"

# ---------------- ENTITY EXTRACTION ----------------
def extract_entities(text, lang):
    ents = {"PERSON": [], "ORG": [], "DATE": [], "MONEY": [], "LOCATION": []}
    if lang == "Hindi":
        ents["PERSON"] = re.findall(r"श्री\s+[अ-ह]+\s*[अ-ह]*", text)
        ents["ORG"] = re.findall(r"([A-Za-zअ-ह]+\s+(?:प्राइवेट लिमिटेड|लिमिटेड|कंपनी))", text)
        ents["DATE"] = re.findall(r"\d{1,2}\s+[अ-ह]+\s+\d{4}", text)
        ents["MONEY"] = re.findall(r"₹\s?\d+|\d+\s?रुपये", text)
    else:
        doc = nlp_en(text)
        for e in doc.ents:
            if e.label_ == "PERSON": ents["PERSON"].append(e.text)
            elif e.label_ == "ORG": ents["ORG"].append(e.text)
            elif e.label_ == "DATE": ents["DATE"].append(e.text)
            elif e.label_ == "MONEY": ents["MONEY"].append(e.text)
            elif e.label_ in ["GPE","LOC"]: ents["LOCATION"].append(e.text)
    return ents

# ---------------- AMBIGUITY ----------------
AMBIGUOUS_WORDS = ["reasonable","as per","may be","उचित","समय समय पर"]
def is_ambiguous(clause):
    return any(w in clause.lower() for w in AMBIGUOUS_WORDS)

# ---------------- RISK ----------------
HIGH = ["indemnify","penalty","terminate","liability","damages","non-compete","intellectual property",
        "क्षतिपूर्ति","दंड","समाप्त","उत्तरदायित्व","प्रतिस्पर्धा","बौद्धिक संपदा"]
MEDIUM = ["arbitration","jurisdiction","lock-in","auto-renew","मध्यस्थता","क्षेत्राधिकार"]
LOW = ["payment","notice","confidentiality","भुगतान","सूचना","गोपनीयता"]

def risk_score(clause):
    t = clause.lower()
    for w in HIGH:
        if w in t: return "High"
    for w in MEDIUM:
        if w in t: return "Medium"
    for w in LOW:
        if w in t: return "Low"
    return "Low"

def obligation_type(clause):
    c = clause.lower()
    if "shall not" in c or "नहीं करेगा" in c: return "Prohibition"
    if "shall" in c or "must" in c or "करेगा" in c: return "Obligation"
    if "may" in c or "सकता है" in c: return "Right"
    return "Neutral"

# ---------------- LLM ----------------
def llm_summarize(text):
    if not USE_LLM:
        return "LLM disabled. Showing rule-based summary."
    prompt = f"Summarize this contract in simple business English:\n{text}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content

def llm_suggest(clause):
    if not USE_LLM:
        return "Consider renegotiating this clause."
    prompt = f"Suggest safer alternative for this clause:\n{clause}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content

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

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

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

    high = 0
    st.subheader("📑 Clause Analysis")

    for i, cl in enumerate(clauses):
        r = risk_score(cl)
        o = obligation_type(cl)
        amb = is_ambiguous(cl)

        css = "risk-low"
        if r == "High":
            css = "risk-high"
            high += 1
        elif r == "Medium":
            css = "risk-medium"

        if amb:
            css += " ambiguous"

        sug = llm_suggest(cl) if r=="High" else "Clause acceptable."

        st.markdown(
            f"<div class='{css}'><b>Clause {i+1} - {r}</b>"
            f"<br><b>Type:</b> {o}"
            f"<br><b>Ambiguous:</b> {amb}"
            f"<br><br>{cl}"
            f"<br><b>Suggestion:</b> {sug}</div>",
            unsafe_allow_html=True
        )
        st.divider()

    st.subheader("📊 Contract Risk")
    if high > 2:
        st.error("Overall Risk: HIGH")
    else:
        st.success("Overall Risk: LOW")

    summary = llm_summarize(text)
    st.subheader("📝 Summary Report")
    st.write(summary)

    if st.button("Export PDF"):
        st.session_state.pdf_path = export_pdf(summary)
        st.success("PDF generated!")

    if st.session_state.pdf_path:
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, "contract_report.pdf", "application/pdf")

    st.subheader("📄 SME-Friendly Contract Templates")
    temp_choice = st.selectbox("Choose Template", list(TEMPLATES.keys()))
    st.text_area("Template Preview", "\n".join(TEMPLATES[temp_choice]), height=200)

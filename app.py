import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import torch
from typing import List
from difflib import ndiff

# Load SBERT model
model = SentenceTransformer('paraphrase-mpnet-base-v2')

st.set_page_config(page_title="PDF Difference Viewer", layout="wide")
st.title("📄 PDF Semantic Difference Viewer")

# Function to extract text from PDF
def extract_text(pdf_file) -> List[str]:
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return [para.strip() for para in text.split("\n") if para.strip()]

# Function to compare text semantically
def compare_texts(text_a: List[str], text_b: List[str], threshold_mod=0.85, threshold_add_del=0.6):
    results = []
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)

    matched_b = set()
    add_count = del_count = mod_count = 0

    for idx_a, a_vec in enumerate(emb_a):
        scores = util.cos_sim(a_vec, emb_b)[0]
        best_match_idx = torch.argmax(scores).item()
        best_score = scores[best_match_idx].item()

        if best_score >= threshold_mod:
            results.append(("modified", text_a[idx_a], text_b[best_match_idx]))
            matched_b.add(best_match_idx)
            mod_count += 1
        elif best_score < threshold_add_del:
            results.append(("removed", text_a[idx_a], ""))
            del_count += 1

    # Find additions
    for idx_b, para_b in enumerate(text_b):
        if idx_b not in matched_b:
            results.append(("added", "", para_b))
            add_count += 1

    return results, add_count, del_count, mod_count

# Streamlit file uploader
col1, col2 = st.columns(2)
with col1:
    pdf1 = st.file_uploader("Upload First PDF", type="pdf")
with col2:
    pdf2 = st.file_uploader("Upload Second PDF", type="pdf")

if pdf1 and pdf2:
    text_a = extract_text(pdf1)
    text_b = extract_text(pdf2)

    st.success("PDFs uploaded and processed. Comparing...")
    results, add_count, del_count, mod_count = compare_texts(text_a, text_b)

    st.subheader("📊 Summary Report")
    st.markdown(f"- ✅ **Added**: {add_count}\n- ❌ **Removed**: {del_count}\n- ✏️ **Modified**: {mod_count}")

    st.subheader("📝 Detailed Comparison")
    for tag, old, new in results:
        if tag == "added":
            st.markdown(f"<div style='background-color:#d4edda;padding:10px;border-radius:5px;'>✅ <b>Added:</b> {new}</div>", unsafe_allow_html=True)
        elif tag == "removed":
            st.markdown(f"<div style='background-color:#f8d7da;padding:10px;border-radius:5px;'>❌ <b>Removed:</b> {old}</div>", unsafe_allow_html=True)
        elif tag == "modified":
            st.markdown(f"<div style='background-color:#fff3cd;padding:10px;border-radius:5px;'>✏️ <b>Modified:</b><br><i>Old:</i> {old}<br><i>New:</i> {new}</div>", unsafe_allow_html=True)
else:
    st.info("Please upload two PDF files to begin comparison.")

# 📄 PDF Difference Highlighter with SBERT (Streamlit App)

This web application allows users to upload **two PDF files**, compare them using **Sentence-BERT (SBERT)** for semantic similarity, and highlight the **added**, **removed**, and **modified** content with color-coded text.

Built with **Streamlit**, it's perfect for tracking changes across PDF versions — even when rephrased!

---

## 🚀 Features

- ✅ Upload two PDF documents side-by-side
- 🧠 Semantic comparison using **SBERT** (`paraphrase-mpnet-base-v2`)
- 🎨 Color-coded highlights:
  - ✅ Green → Added text
  - ❌ Red → Removed text
  - ✏️ Yellow → Modified (rephrased) text
- 📊 Summary report showing number of additions, deletions, and modifications
- 💡 Intuitive and clean Streamlit interface

---

## 📂 How to Use

1. **Clone the repo or download the files**

2. **Install dependencies:**
   pip install -r requirements.txt

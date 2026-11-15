import streamlit as st
from pypdf import PdfReader

st.title("BIT Study Tutor")

uploaded = st.file_uploader("Upload a PYQ or notes PDF", type="pdf")

if uploaded:
    reader = PdfReader(uploaded)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    st.write(f"Read {len(reader.pages)} pages, {len(text)} characters")

    if len(text.strip()) == 0:
        st.warning("No text found. This PDF may be a scan (images only). Try a different PDF.")
    else:
        st.text_area("Extracted text", text, height=300)
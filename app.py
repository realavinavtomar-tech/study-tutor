import io
import os
import re

import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash"

STOPWORDS = {
    "explain", "about", "what", "which", "describe", "write", "give",
    "between", "difference", "with", "from", "that", "this", "their",
    "each", "main", "list", "state", "define", "discuss", "topic",
}


def ask(prompt):
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text


@st.cache_data(show_spinner="Reading PDF...")
def read_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    return [(p.extract_text() or "") for p in reader.pages]


def find_pages(pages, question, k=10):
    words = set(re.findall(r"[a-z]{4,}", question.lower())) - STOPWORDS
    scores = [
        (sum(text.lower().count(w) for w in words), i)
        for i, text in enumerate(pages)
    ]
    best = sorted(scores, reverse=True)[:k]
    return sorted(i for score, i in best if score > 0)


st.title("BIT Study Tutor")
st.caption("AI answers can be wrong, especially in maths. Always double-check.")

col_a, col_b = st.columns([1, 3])
with col_a:
    if os.path.exists("me.jpg"):
        st.image("me.jpg", width=200)
with col_b:
    st.markdown(
        "**Built by Avinav Tomar** · 1st-year CSE, BIT Mesra  \n"
        "[GitHub](https://github.com/realavinavtomar-tech) · "
        "[LinkedIn](https://www.linkedin.com/in/avinav-tomar-973152299/)"
    )

uploaded = st.file_uploader("Upload a PYQ or notes PDF", type="pdf")
pages = []
if uploaded:
    pages = read_pdf(uploaded.getvalue())
    st.success(f"Read {len(pages)} pages")

mode = st.radio(
    "Which pages should I use?",
    ["Auto-find relevant pages", "Choose page range"],
    horizontal=True,
)

start, end = 1, 1
if pages and mode == "Choose page range":
    start = st.number_input("From page", 1, len(pages), 1)
    end = st.number_input("To page", start, len(pages), min(len(pages), start + 14))

question = st.text_area(
    "Question or topic (for example: causes of water pollution)"
)

c1, c2, c3 = st.columns(3)
action = None
if c1.button("Explain"):
    action = "explain"
if c2.button("Generate 10 MCQs"):
    action = "mcq"
if c3.button("Important topics"):
    action = "topics"

if action:
    if not pages:
        st.warning("Upload a PDF first.")
        st.stop()

    if mode == "Auto-find relevant pages":
        if not question.strip():
            st.warning("Type a question or topic so I can find the right pages.")
            st.stop()
        idx = find_pages(pages, question)
        if not idx:
            st.warning(
                "Couldn't find matching pages. Try different words, "
                "or use 'Choose page range'."
            )
            st.stop()
    else:
        idx = list(range(start - 1, end))

    material = "\n\n".join(f"[Page {i + 1}]\n{pages[i]}" for i in idx)
    st.caption("Using pages: " + ", ".join(str(i + 1) for i in idx))

    rules = (
        "You are a tutor for BIT Mesra B.Tech students. Use only the study "
        "material below. Start directly, with no greeting. Use short "
        "headings and simple language. If the material doesn't cover "
        "something, say so clearly.\n\n"
    )

    if action == "explain":
        task = f"Explain step by step: {question}"
    elif action == "mcq":
        task = (
            "Create 10 multiple-choice questions with 4 options each, mark "
            "the correct answer, and add a one-line explanation for each."
        )
    else:
        task = (
            "List the 8 most important topics for an exam, ranked, with a "
            "one-line study tip for each."
        )

    with st.spinner("Thinking..."):
        try:
            st.write(ask(f"{rules}STUDY MATERIAL:\n{material[:40000]}\n\nTASK:\n{task}"))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
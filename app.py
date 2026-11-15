import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash"


def ask(prompt):
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text


st.title("BIT Study Tutor")
st.caption("AI answers can be wrong, especially in maths. Always double-check.")

uploaded = st.file_uploader("Upload a PYQ or notes PDF", type="pdf")
text = ""

if uploaded:
    reader = PdfReader(uploaded)
    total = len(reader.pages)
    st.success(f"Read {total} pages")

    start = st.number_input("From page", 1, total, 1)
    end = st.number_input("To page", start, total, min(total, start + 14))

    text = "\n".join(
        reader.pages[i].extract_text() or "" for i in range(start - 1, end)
    )

question = st.text_area("Paste a question you want explained")

if st.button("Explain") and question:
    with st.spinner("Thinking..."):
        try:
            answer = ask(
                "You are a friendly tutor for BIT Mesra B.Tech students. "
                "Explain the answer step by step in simple language.\n\n"
                f"STUDY MATERIAL:\n{text[:30000]}\n\nQUESTION:\n{question}"
            )
            st.write(answer)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
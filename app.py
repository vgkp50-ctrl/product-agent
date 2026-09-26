import streamlit as st

from transformers import pipeline
from retriever import retrieve
from dotenv import load_dotenv
load_dotenv()

@st.cache_resource
def load_model():

    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-3B-Instruct",
        device_map="auto"
    )

llm = load_model()

st.title(
    "Product Catalog AI Assistant"
)

question = st.chat_input(
    "Ask about products"
)

if question:

    context = retrieve(question)

    prompt = f"""
Catalog:
{context}

Question:
{question}

Answer from catalog only.
"""

    result = llm(
        prompt,
        max_new_tokens=150
    )

    st.write(
        result[0]["generated_text"]
    )
import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import time

import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
set_background("default.jpg")    

# -------- Load index + metadata + chunks --------
INDEX_PATH = "data/relativity_index.faiss"
META_PATH = "data/relativity_meta.json"
CHUNKS_PATH = "data/relativity_chunks.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
LLM_MODEL = "meta-llama/llama-3.3-8b-instruct:free"

# Load FAISS + data
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r") as f:
    chunk_meta = json.load(f)
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

embedder = SentenceTransformer(EMBED_MODEL)

# -------- Streamlit UI --------
st.set_page_config(page_title="Relativity Q&A", layout="wide")
st.title("Special Relativity Q&A")
st.write("Ask a question about the document and get an answer from the retrieved context.")

# API key from secrets
api_key = st.secrets["OPENROUTER_API_KEY"]
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

def search(query, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append({
            "text": chunks[idx],
            "meta": chunk_meta[idx],
            "score": float(score)
        })
    return results

def build_prompt(query, retrieved):
    context = "Extra Knowledge:\nEinstein published relativity in 1905 Albert Einstein introduced special relativity in 1905 .\n\n"
    for r in retrieved:
        src = os.path.basename(r["meta"].get("source", "unknown"))
        page = r["meta"].get("page", "?")
        context += f"[Source: {src}, page {page}]\n{r['text']}\n\n---\n\n"

    return f"""
You are a helpful assistant. Answer the following question using ONLY the provided Context.
If the answer cannot be found, say "NOT IN DOCUMENTS".

Context:
{context}

Question: {query}
Answer:
"""

def ask_question(query):
    start = time.time()
    retrieved = search(query, top_k=5)
    prompt = build_prompt(query, retrieved)

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
        
        
    )
    answer = completion.choices[0].message.content
    latency = round(time.time() - start, 3)

    return answer, retrieved, latency

query = st.text_input("Enter your question:")
if query:
    with st.spinner("Searching and generating answer..."):
        answer, retrieved, latency = ask_question(query)
    st.subheader("Answer")
    st.write(answer)
    st.caption(f"⏱️ Latency: {latency}s")

    with st.expander("🔎 Retrieved Context"):
        for r in retrieved:
            st.markdown(f"**Page {r['meta']['page']}** — Score: `{r['score']:.4f}`")
            st.write(r['text'])
            st.markdown("---")











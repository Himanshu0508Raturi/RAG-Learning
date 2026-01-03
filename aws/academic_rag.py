import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

load_dotenv()

# ------------------------------------
# Load embedding model ONCE
# ------------------------------------
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L12-v2",
    device="cpu"
)

# ------------------------------------
# Pinecone Retrieval
# ------------------------------------
def retrieve_query(query, k=3):
    query_embedding = embedding_model.encode(query).tolist()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(host=os.getenv("PINECONE_HOST_ACAD"))

    results = index.query(
        vector=query_embedding,
        top_k=k,
        namespace="rag-docs",
        include_metadata=True
    )

    return results

# ------------------------------------
# LLM Response
# ------------------------------------
def llm_response(query, retrieval_result, top_k=3):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024
    )

    res = retrieval_result.matches[:top_k]

    context_chunks = [
        match.metadata.get("text", "")
        for match in res
    ]

    if not context_chunks:
        return "No relevant context found to answer the question."

    context = "\n\n".join(context_chunks)

    prompt = f"""
Use the following context to answer the question concisely.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content

# ------------------------------------
# Streamlit UI
# ------------------------------------
def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("📄 Chat with PDF")

    user_question = st.text_input("Ask a question from the PDF files")

    if st.button("Output") and user_question.strip():
        with st.spinner("Processing..."):
            retrieval_result = retrieve_query(user_question)
            answer = llm_response(user_question, retrieval_result)
            st.write(answer)
            st.success("Done")

if __name__ == "__main__":
    main()

## deploying agentic_rag.py
import streamlit as st
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from typing import TypedDict, List
from langgraph.graph import StateGraph , END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_core.documents import Document

load_dotenv()
class AgentState(TypedDict):
    question: str
    documents: List[str]
    ans: str
    needs_retrieval : bool

## agentic RAG
llm = ChatGroq(groq_api_key= os.getenv("NEW_GROQ_API_KEY") , model_name="llama-3.1-8b-instant",temperature=0.1,max_tokens=1024)

def decide_retrieval(state: AgentState) -> AgentState:
    """
    Decide if we need to retrieve documents based on the question
    """
    question = state["question"]
    
    prompt = f"""
You are an AI controller in a Retrieval-Augmented Generation (RAG) system.

Task:
Decide whether external knowledge retrieval is REQUIRED to answer the user query correctly and confidently.

Rules:
- Choose RETRIEVE if the query:
  - Refers to specific documents, standards, datasets, or user-provided files
  - Requires precise, technical, or domain-specific information
  - Cannot be answered confidently without external sources
- Choose GENERATE if the query:
  - Is conversational or opinion-based
  - Can be answered accurately using general knowledge alone

Important:
- If you are uncertain, choose RETRIEVE.

User Query:
"{question}"

Respond with ONLY one word:
RETRIEVE or GENERATE

"""
    response = llm.invoke(prompt)
    decision = response.content.strip().upper()
    if decision not in {"RETRIEVE", "GENERATE"}:
      decision = "RETRIEVE"  # safe default
    needs_retrieval = decision == "RETRIEVE"
    return {**state, "needs_retrieval": needs_retrieval}

def retrieve_document(state: AgentState) ->AgentState:
    """
    Retrieve relevant documents based on the question
    """
    question = state["question"]
    # Pinecone Configuration
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "academic-rag"
    
    host = os.getenv("PINECONE_HOST_ACAD")
    index = pc.Index(host= host)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    query_embedding = model.encode(question).tolist()
    matching_result = index.query(
        vector=query_embedding,
        top_k=3,
        namespace="rag-docs",
        include_metadata=True
    )

    context = []
    for doc in matching_result.matches:
        context.append(doc.metadata['text'])
    if not context:
        return "No relevant context found to answer the question."
    
    return {**state, "documents": context}

def generate_answer(state: AgentState) -> AgentState:
    """
    Generate an answer using the retrieved documents or direct response
    """
    question = state["question"]
    context = state["documents"]

    if context:
        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {question}

Answer:"""
    else:
        prompt = f"Answer the following question: {question}"
    
    response = llm.invoke(prompt)
    answer = response.content
    return {**state, "ans": answer}

def should_retrieve(state: AgentState) ->str:
     """
    Determine the next step based on retrieval decision
    """
     if state["needs_retrieval"]:
          return "retrieve"
     else:
          return "generate"
     
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("decide",decide_retrieval)
workflow.add_node("retrieve",retrieve_document)
workflow.add_node("generate",generate_answer)

# Set entry point
workflow.set_entry_point("decide")

# Add conditional edges
workflow.add_conditional_edges(
    "decide",
    should_retrieve,
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

# Add edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()

def ask_question(question: str):
    """
    Helper function to ask a question and get an answer
    """
    initial_state = {
        "question": question,
        "documents": [],
        "answer": "",
        "needs_retrieval": False
    }
    
    result = app.invoke(initial_state)
    return result

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("📄 Chat with PDF- Agentic RAG")

    user_question = st.text_input("Ask a question from the PDF files")

    if st.button("Output") and user_question.strip():
        with st.spinner("Processing..."):
            answer = ask_question(user_question)
            st.write(answer['ans'])
            st.success("Done")

if __name__ == "__main__":
    main()
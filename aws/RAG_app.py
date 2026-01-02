import boto3
import streamlit as st
# Using Titan Embedding model to generate embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

# Data Ingestion Libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter # For splitting and chunking data
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader # For converting pdf docs to Document Format

# Vector Embedding and vector Store
from langchain_community.vectorstores import FAISS

# LLM models
from langchain_community.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Bedrock Client - Using Titan Embeddings G1 - Text By: Amazon Embedding model
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("research_data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss= FAISS.from_documents(
        docs,
        bedrock_embedding
    )
    vectorstore_faiss.save_local("faiss_index")

# LLm model = Llama 3.2 1B Instruct By: Meta
def get_claude_llm():
    llm = Bedrock(model_id="meta.llama3-2-1b-instruct-v1:0" , client=bedrock, model_kwargs={'maxTokens':512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but summarize with 
at least 250 words with detailed explanations. 
If you don't know the answer, just say that you don't know.

<context>
{context}
</context>

Question: {input}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template,input_variables=["context","question"]
)

#Response Part
def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


docs = data_ingestion()
get_vector_store(docs)
def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    # with st.sidebar:
    #     st.title("Update Or Create Vector Store:")
        
    #     if st.button("Vectors Update"):
    #         with st.spinner("Processing..."):
    #             docs = data_ingestion()
    #             get_vector_store(docs)
    #             st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embedding)
            llm=get_claude_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
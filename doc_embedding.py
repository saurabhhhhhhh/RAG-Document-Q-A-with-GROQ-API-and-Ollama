import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')

llm=ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")


prompt=PromptTemplate.from_template(
    """
answer the qns based on the provided context only.
please provide the mosta accurate answer based on qns
<context>
{context}
<context>
Question:{input}
"""
)

def create_vector_embedding():
    if "vector" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader=PyPDFDirectoryLoader("rs_paper")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.documents[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
st.title("Document Embedding with Groq")
user_prompt=st.text_input("enter your query from the file")
if st.button("document embedding"):
    create_vector_embedding()
    st.write("vector db is ready")


import time


if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retriever_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])

    with st.expander("document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------")


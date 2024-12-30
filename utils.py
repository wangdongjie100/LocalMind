import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import PyPDFLoader

def retriver_prepare(contents, k=2):
    vector_store = SKLearnVectorStore.from_documents(contents,embedding=NomicEmbeddings(model="nomic-embed-text-v1", inference_mode="local"))
    retriever = vector_store.as_retriever(k=k)
    return retriever

def file_load(file_path):
    pages = []
    _, ext = os.path.splitext(file_path)
    if ext[1:] == "pdf":
        loader = PyPDFLoader(file_path)
        for page in loader.lazy_load():
            pages.append(page)
    else:
        print("Sorry, the uploaded file is not supported!")
    return pages
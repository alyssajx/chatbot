import streamlit as st
import os 
from mistralai import Mistral
from langchain_mistralai import MistralAIEmbeddings
from mistralai.client import MistralClient
import numpy as np


from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

client = MistralClient(api_key=st.secrets["MISTRAL_AI_KEY"])

def load_documents(): 
    document_loader = NotionDirectoryLoader("notion_content")
    return document_loader.load()

#Split content into small chunks 
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["#","##", "###", "\\n\\n","\\n","."],
        chunk_size=1500,
        chunk_overlap=100)

    return text_splitter.split_documents(documents)

def embed(input: str):
    return client.embeddings("mistral-embed", input = input).data[0].embedding

curr = load_documents()
chunks = split_documents(curr)

embeddings = np.array([embed(chunk) for chunk in chunks])
dimension = embeddings.shape[1]

#print(docs) 
print("FAISS updated")
#Embed the content 


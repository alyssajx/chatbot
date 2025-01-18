import streamlit as st
import os 
from mistralai import Mistral
from langchain_mistralai import MistralAIEmbeddings


from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


#Key is not supposed to be public 
#model = "open-mistral-nemo"

#client = Mistral(api_key=api_key)

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


#print(docs) 

#Embed the content 


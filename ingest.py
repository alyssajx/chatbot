import streamlit as st
import os 
from mistralai import Mistral
from langchain_mistralai import MistralAIEmbeddings


from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

#api_key = os.environ["I83wXBlRb3QSe9AMgiYZ1Hekg9WWWuBY"] #KeyError Here 
#Key is not supposed to be public 
#model = "open-mistral-nemo"

#client = Mistral(api_key=api_key)

def load_documents(): 
    document_loader = NotionDirectoryLoader("notion_content")
    documents = document_loader.load()
    return documents

curr = load_documents()


#Split content into small chunks 

text_splitter = RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\\n\\n","\\n","."],
    chunk_size=1500,
    chunk_overlap=100)

each = text_splitter.split_documents(curr)
print(each[1])


#print(docs) 

#Embed the content 


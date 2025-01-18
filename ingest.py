import streamlit as st
import os 
import numpy as np
import pandas as pd
import faiss

#langchain imports
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document

from openai import OpenAI

client = OpenAI()

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

#Embed documents into vector space
curr = load_documents()

chunks = split_documents(curr)

#Convert chunks into vectors embeddings  
def get_embedding(text, model='text-embedding-ada-002'):
    return client.embeddings.create(input=text, model=model)

# Extract text content from the chunks
text_chunks = [chunk.page_content for chunk in chunks]
embeddings_array = np.array(get_embedding(text_chunks)).astype('float32')







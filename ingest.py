import streamlit as st
import os 
import numpy as np
import pandas as pd


from langchain_community.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document

from openai import OpenAI

client = OpenAI()

#completion = client.chat.completions.create(
 # model="gpt-4o-mini",
  #store=True,
  #messages=[
   # {"role": "user", "content": "write a haiku about ai"}
  #],
  #max_tokens = 100
#)'''

#print(completion.choices[0].message);

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

# Extract text content from the chunks
text_chunks = [chunk.page_content for chunk in chunks]

#Convert chunks into vectors embeddings 
embeddings = client.embeddings.create(input=text_chunks, model='text-embedding-ada-002')
#db = FAISS.from_documents(chunks, embeddings)
#db.save_local("faiss_index")





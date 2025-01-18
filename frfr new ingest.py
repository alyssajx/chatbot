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
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

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
    response = openai.embeddings.create(input = text, model = model)
    return response.data[0].embedding

# Extract text content from the chunks
text_chunks = [chunk.page_content for chunk in chunks]
embeddings_list = [get_embedding(chunk) for chunk in text_chunks]
embeddings_array = np.array(embeddings_list).astype('float32')
embedding_dim = 1536
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings_array)

#query
question = input("Please ask a question: ")

def check_for_sensitive_questions(question):
    sensitive_keywords = ["who is the killer", "who killed simon", "who murdered simon", "who is the murderer", "is nate the killer", 
                          "is addy the killer", "is cooper the killer", 'is bronwyn the killer', 'tell me the killer', 'tell me the murderer',
                          "did nate kill simon", 'did nate murder simon',"did addy kill simon", 'did addy murder simon', "did cooper kill simon", 
                          'did cooper murder simon', "did bronwyn kill simon", 'did bronwyn murder simon', 'killer', 'murderer','arrested','culprit','suspect',
                          'caught']
    if any(keyword.lower() in question.lower() for keyword in sensitive_keywords):
        return "Sorry, I can't tell you that."
    return None

sensitive_response = check_for_sensitive_questions(question)
if sensitive_response:
    print(sensitive_response)
else: 
    question_embeddings = np.array([get_embedding(question)]).astype('float32')

    #retrieval 
    D, I = index.search(question_embeddings, k=2) 
    distance_threshold = 0.5
    def retrieval():
        if D[0][0] > distance_threshold:
            return "Sorry, I don't know the answer."
        else: 
            retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
            print("Retrieved Chunks:", retrieved_chunk)
        
def load_chain():
    """
    The `load_chain()` function initializes and configures a conversational retrieval chain for
    answering user questions.
    :return: The `load_chain()` function returns a ConversationalRetrievalChain object.
    """
    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key)

    # Define the paths
    folder_path = "faiss_index"
    index_file_path = "faiss_index/index.faiss"

    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

    # Generate a new FAISS index
    documents = load_documents()  # Replace this with your document-loading function
    chunks = split_documents(documents)  # Replace this with your chunk-splitting function
    faiss_index = FAISS.from_documents(chunks, embeddings)
    faiss_index.save_local(folder_path)

    # Load our local FAISS index as a retriever
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Create memory 'chat_history'
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")

    # Create system prompt
    template = """
    You are an AI assistant for answering questions about the Blendle Employee Handbook.
    You are given the following extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say 'Sorry, I don't know ... 😔.'
    Don't try to make up an answer.
    If the question is not about the Blendle Employee Handbook, politely inform them that you are tuned to only answer questions about the Blendle Employee Handbook.
    
    {context}
    Question: {question}
    Helpful Answer:"""

    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True
    )

    # Add system prompt to chain
    # Can only add it at the end for ConversationalRetrievalChain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)

    return chain


# Initialize the chain
chain = load_chain()
print(chain)









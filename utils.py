import numpy as np
import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

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
    question_embeddings = np.array([get_embedding(question)])

    #retrieval 
    D, I = index.search(question_embeddings, k=2) 
    distance_threshold = 0.7
    def retrieval():
        if D[0][0] > distance_threshold:
            return "Sorry, I don't know the answer."
        else: 
            retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
            return retrieved_chunk
        
def load_chain():
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature = 0)
    vector_store = FAISS.load_local("faiss_index", embeddings)
	retriever = vector_store.as_retriever(search_kwargs={"k": 3})
		
	# Create memory 'chat_history' 
	memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")
		
	# Create system prompt
	template = """
    You are an AI assistant for answering questions about the Blendle Employee Handbook.
    You are given the following extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say 'Sorry, I don't know ... 😔. 
    Don't try to make up an answer.
    If the question is not about the Blendle Employee Handbook, politely inform them that you are tuned to only answer questions about the Blendle Employee Handbook.
    
    {context}
    Question: {question}
    Helpful Answer:"""
		
	# Create the Conversational Chain
	chain = ConversationalRetrievalChain.from_llm(llm=llm, 
				                                          retriever=retriever, 
				                                          memory=memory, 
				                                          get_chat_history=lambda h : h,
				                                          verbose=True)
		
	# Add systemp prompt to chain
	# Can only add it at the end for ConversationalRetrievalChain
	QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
	chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)
		
	return chain
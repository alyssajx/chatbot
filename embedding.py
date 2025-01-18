from mistralai.client import MistralClient
import numpy as np

client = MistralClient(api_key = )

def embed(input: str):
    return client.embeddings("mistral-embed", input = input).data[0].embedding

embeddings = np.array([embed(chunk) for chunk in chunks])
dimension - embeddings.shape[1]

#query
question = input("Please ask a question:")
question_embeddings = np.array([embed(question)])

#retrieval 
D, I = index.search(question_embeddings, k=2) 
distance_threshold = 0.7
def retrieval():
    if D[0][0] > distance_threshold:
        return "Sorry, I don't know the answer."
    else: 
        retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
        return retrieved_chunk
        
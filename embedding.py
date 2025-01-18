from mistralai.client import MistralClient
import numpy as np

client = st.secrets["MISTRAL_AI_KEY"]

def embed(input: str):
    return client.embeddings("mistral-embed", input = input).data[0].embedding

embeddings = np.array([embed(chunk) for chunk in chunks])
dimension = embeddings.shape[1]

import faiss
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
        


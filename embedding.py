from mistralai.client import MistralClient
import numpy as np

client = MistralClient(api_key = )

def embed(input: str):
    return client.embeddings("mistral-embed", input = input).data[0].embedding

embeddings = np.array([embed(chunk) for chunk in chunks])
dimension - embeddings.shape[1]
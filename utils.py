import numpy as np

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
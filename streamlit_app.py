import streamlit as st
import time

# Show title and description.
st.title("💬 Ask me how I DIED")

st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "You have 20 (can change) prompts to solve the mystery. Solve who killed Simon.\n"
    "The fewer prompts you use, the better - Try your best!\n\n"
    "5 students enter detention, but only 4 make it out alive. Simon Kelleher, the creator of Bayview High’s infamous gossip app About That, dies under suspicious circumstances, leaving Bronwyn, the overachiever; Nate, the criminal; Cooper, the athlete; and Addy, the beauty queen, as prime suspects. Simon had dirt on all of them—secrets that could ruin their lives—and his death might not have been an accident. Who killed Simon?\n"
    "Here are some examples you might consider using:\n"
    "- Where was Addy during the incident?\n"
    "- Did Nate have any reason to kill Simon?\n"
    "- How did Simon die?"
)

company_logo = 'https://i.postimg.cc/8PrMgd06/spy.png'

# Initialize LLM chain
chain = load_chain() #loadchain not defined

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": "Hi, I am the chatbot. Ask me a question!"}]


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=company_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# Chat logic
if query := st.chat_input("Ask about the murderer"):
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=company_logo):
        message_placeholder = st.empty()
        # Send user's question to our chain
        result = chain({"question": query})
        response = result['answer']
        full_response = ""

    # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": response})
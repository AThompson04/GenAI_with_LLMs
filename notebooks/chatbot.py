from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage,HumanMessage,AIMessage
import streamlit as st
import time
import os

def stream_data(text):
    for word in text.split():
        yield word + ' '
        time.sleep(0.05)

def disable():
    st.session_state['disabled'] = True

if __name__ == '__main__':
    st.set_page_config(
        page_title='Chatbot',
        page_icon='../assets/catworking.png',
        layout = 'wide'
    )

    st.subheader('Your Custom Chatbot')

    # Initialising chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # System Message editability
    if 'disabled' not in st.session_state:
        st.session_state['disabled'] = False

    with st.sidebar:
        openai_key = st.text_input(label = 'OpenAI API Key', type = 'password')
        # Choosing the Model Type
        model = st.segmented_control(label='OpenAI Model', options=['gpt-4o', 'gpt-3.5-turbo', 'gpt-4o-mini'],
                                     default='gpt-3.5-turbo', selection_mode='single',
                                     help='This model will be used to summarise you document or text.')
        if not model:
            model = 'gpt-3.5-turbo'
            st.warning('No model has been selected, so gpt-3.5-turbo will be used by default')
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
            chat = ChatOpenAI(model_name=model, temperature=0.5)

        system_message = st.text_input(label = 'System Role', disabled = st.session_state.disabled)

        if system_message:
            if not any(isinstance(x, SystemMessage) for x in st.session_state.messages): # checking to see if there are any messages on the list. If there are none (because of the not) then add the system message
                st.session_state.messages.append(
                    SystemMessage(content = system_message)
                ) # You can only do this once per session (Adding the system message to the chat history)
                st.session_state['disabled'] = True

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            role = 'user'
        elif isinstance(message, AIMessage):
            role = 'assistant'
        else:
            continue  # Skip system messages for display
        with st.chat_message(role):
            st.markdown(message.content)

    if openai_key:
        # Accept user input
        if prompt := st.chat_input('Message Chatbot'):
            # Add user message to chat history
            st.session_state.messages.append(HumanMessage(content = prompt))

        # If the user has asked a question but did not add a System Message => Add a default System Message
            if len(st.session_state.messages) >= 1:
                if not isinstance(st.session_state.messages[0], SystemMessage):  # If the first message is not a system message then add one
                    st.session_state.messages.insert(0, SystemMessage(content='You are a helpful assistant.'))
                    st.session_state['disabled'] = True

            # Display user message in chat message container
            with st.chat_message('user'):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message('assistant'):
                ai_response = chat.invoke(st.session_state.messages)
                response = st.write_stream(stream_data(ai_response.content))

            # Add AI response to chat history
            st.session_state.messages.append(AIMessage(content = ai_response.content))
## Importing Libraries
import streamlit as st
import time
from langchain_openai import OpenAIEmbeddings

## Functions
def load_document(file):
    """ A file loader that converts PDFs and other document formats into LangChain documents.

    Parameter
    ---------
    'file': The file path or URL.
        The file name of the document stored on your PC or the URL of an online document.

    Return
    ---------
    'data': List.
        A list of LangChain documents. There will be a document for each page in the document.
    """
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)  # You can load an online PDF with a URL
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print("Document format is not supported :(")
        return None

    data = loader.load()  # This will return a list of LangChain Documents, one document for each page in the PDF.
    return data


def chunk_data(data, chunk_size=256, chunk_overlap = 20):
    """
    The data passed into this function will be split into chunks of data.

    Parameters
    ----------
    'data': LangChain Document Format Data.
        Data that was converted from a document into LangChain document format.
    'chunk_size': Numeric.
        The number of characters in each chunk. The default is 256.

    Returns
    ----------
    'chunks': List.
        A list of chunks.
    """

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings_chroma(chunks, file_name):
    import os
    import chromadb
    from langchain_chroma import Chroma
    from chromadb.config import Settings

    # Create a unique directory for the file
    persist_dir = os.path.join("./chroma_storage", f"{file_name.split('.')[0]}_{int(time.time())}")
    if not os.path.exists(persist_dir):
        #st.write(f"Directory does not exist. Creating {persist_dir}")
        os.makedirs(persist_dir, exist_ok=True)
    #else:
        #st.write(f"Directory {persist_dir} already exists.")

    try:
        client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
    except ValueError as e:
        #st.write(f"Error initializing PersistentClient: {e}")
        return None

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    # Create the vector store with the client
    collection_name = 'my_vector_store'
    try:
        #st.write(f"Attempting to create or fetch the collection: {collection_name}")
        collections = client.get_or_create_collection(collection_name)
    except ValueError as e:
        #st.write(f"Error creating/fetching collection: {e}")
        return None

    # Prepare the unique document IDs
    text_chunks = [chunk.page_content for chunk in chunks]
    ids = [f"doc_{i}" for i in range(len(text_chunks))]
    embedding_vectors = embeddings.embed_documents(text_chunks)

    # Add the documents to the collection
    try:
        #st.write(f"Adding {len(text_chunks)} documents to the collection.")
        collections.add(
            ids=ids,
            embeddings=embedding_vectors,
            documents=text_chunks
        )
    except Exception as e:
        #st.write(f"Error adding documents to collection: {e}")
        return None

    try:
        #st.write(f"Creating Chroma vector store at {persist_dir}")
        vector_store = Chroma(
            collection_name=collection_name,
            client=client,
            embedding_function=embeddings,
        )
        #st.write(f"Vector store created at {persist_dir}")
        return vector_store
    except Exception as e:
        #st.write(f"Error creating Chroma vector store: {e}")
        return None

def ask_with_memory(vector_store, k, question, chat_history):
    from langchain.chains import ConversationalRetrievalChain
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={'k': k}, search_type='similarity')

    system_template = r'''
    Use the following pieces of context to answer the user's question.
    ---------------
    Context: ```{context}```
    If the context provided does not contain information about the question,then 
    use your own knowledge outside the context
    '''

    # Leave the last two lines of the system_template out if you only want it to answer question within the context of the document

    user_template = '''
    Question: ```{question}```
    Chat History: ```{chat_history}```
    '''

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)

    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Link the ChatGPT LLM
        retriever=retriever,  # Link the vector store based retriever
        chain_type='stuff',  # This means that it must use all the text from the documents when analysing
        combine_docs_chain_kwargs={'prompt': qa_prompt},
        verbose=False
    )
    result = crc.invoke({'question': question, 'chat_history': chat_history})
    return result

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens/1000*0.0004

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

if __name__ == '__main__':
    import os

    st.set_page_config(page_title='RAG',
                       page_icon='../assets/catworking.png',
                       layout='wide')
    st.subheader('Question-Answering Application for Private Documents')
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'txt', 'docx'])
        chunk_size = st.number_input('Chunk Size:', min_value=100, max_value=2048, value=512)
        k = st.number_input('k', min_value=1, max_value=20, value=3)
        add_data = st.button('Add Data')

        if uploaded_file and add_data and api_key:
            with st.spinner('Reading, chunking and embedding file...'):
                if 'vs' in st.session_state:
                    del st.session_state['vs']
                st.session_state.history = []
                st.session_state.messages = []

                # Clear the entire Chroma storage directory
                import shutil
                persist_dir = "./chroma_storage"
                if os.path.exists(persist_dir):
                    shutil.rmtree(persist_dir)  # Deletes the entire Chroma storage directory

                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding Cost: ${embedding_cost:.4f}')

                vector_store = create_embeddings_chroma(chunks, file_name)

                # Saving the vector store in the session state so that this does not rerun constantly
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully')
        elif uploaded_file and add_data and not api_key:
            st.warning('Enter your OpenAI API key.')

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Accept user input
    if q:= st.chat_input('Say Something'):
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs

            # Generate Response
            result = ask_with_memory(vector_store, k, q, st.session_state.history)

            # Add question and answer to the history (given to the AI)
            st.session_state.history.append((q, result['answer']))

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": q})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(q)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                ai_response = st.write_stream(response_generator(result['answer']))

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": ai_response})



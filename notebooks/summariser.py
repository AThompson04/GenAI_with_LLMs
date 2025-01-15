## Importing Libraries
import os
import time

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

## API Key - Remove from Final Code
os.environ['OPENAI_API_KEY'] = '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'

## Defining Functions
def load_document(file):
    """ A file loader that converts PDFs and other document formats into LangChain documents.

    Parameter
    ---------
    'file': The file path or URL.
        The file name of the document stored on your PC or the URL of an online document.

    Return
    ---------
    'data': DATA TYPE
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

    # Add elif statement for all other document formats that you would like to convert.

    else:
        return None

    data = loader.load()
    return data

def clear_document():
    st.session_state.document = ''

def stream_data(text):
    for word in text.split():
        yield word + ' '
        time.sleep(0.05)

def short_doc_no_instruction(llm, docs):
    prompt_template = '''
    You are an expert copywriter with expertise in summarising documents. Write a summary of the following text.
    TEXT: `{text}`
    '''
    prompt = PromptTemplate(
        input_variables = ['text'],
        template = prompt_template
    )

    chain = load_summarize_chain(
        llm = llm,
        chain_type = 'stuff',
        prompt = prompt,
        verbose = False
    )
    output_summary = chain.invoke({'input_documents':docs})
    return output_summary['output_text']

def short_doc_lang(llm, docs, translate_to):
    template = '''
    Write summary of the following text:
    TEXT: `{text}`
    Translate the summary to {language}.
    '''

    prompt = PromptTemplate(
        input_variables = ['text', 'language'],
        template = template
    )

    chain = load_summarize_chain(
        llm = llm,
        chain_type = 'stuff',
        prompt = prompt,
        verbose = False
    )
    summary = chain.invoke({'input_documents': docs, 'language': translate_to})
    return summary['output_text']

def short_doc_instructions(llm, docs, instructions):
    prompt_template = '''
    `{instruction}`.
    TEXT: `{text}`
    '''
    prompt = PromptTemplate(
        input_variables = ['instruction','text'],
        template = prompt_template
    )

    chain = load_summarize_chain(
        llm = llm,
        chain_type ='stuff',
        prompt = prompt,
        verbose = False
    )
    output_summary = chain.invoke({'input_documents': docs, 'instruction': instructions})
    return output_summary['output_text']

def long_doc_map_no_instructions(llm, docs, chunk_size = 100, chunk_overlap = 5):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.create_documents([docs])

    chain = load_summarize_chain(
        llm = llm,
        chain_type = 'map_reduce',
        verbose = False
    )
    output_summary = chain.invoke(chunks)
    return output_summary['output_text']

def long_doc_map_prompt(llm, docs, final_prompt, chunk_size = 100, chunk_overlap = 5):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.create_documents([docs])

    # First Prompt (Map Prompt)
    map_prompt = '''
    Write a concise summary of the following:
    TEXT: `{text}`
    CONCISE SUMMARY:
    '''

    map_prompt_template = PromptTemplate(
        input_variables = ['text'],
        template = map_prompt
    )

    # Second Prompt (Combine Prompt)
    combine_prompt = '''
    `{combine}`
    TEXT: `{text}`
    '''

    combine_prompt_template = PromptTemplate(
        input_variables = ['text'],
        template = combine_prompt
    )

    summary_chain = load_summarize_chain(
        llm = llm,
        chain_type = 'map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = combine_prompt_template,
        verbose = False
    )
    output = summary_chain.invoke({'input_documents': chunks, 'combine': final_prompt})
    return output['output_text']

def long_doc_refine_no_instructions(llm, docs, chunk_size = 100, chunk_overlap = 5):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.create_documents([docs])

    chain = load_summarize_chain(
        llm = llm,
        chain_type = 'refine',
        verbose=False
    )
    output_summary = chain.invoke(chunks)
    return output_summary['output_text']

def long_doc_refine_prompt(llm, docs, final_prompt, chunk_size = 100, chunk_overlap = 5):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.create_documents([docs])

    prompt_template = '''
    Write a concise summary of the following extraxting the key information:
    Text: `{text}
    CONCISE SUMMARY: 
    '''

    initial_prompt = PromptTemplate(
        template = prompt_template,
        input_variables = ['text']
    )

    refine_template = '''
    Your job is to provide a final summary.
    I have provided an existing summary to a certain point: `{existing_answer}`.
    Please refine the existing summary with more context below.
    ------
    `{text}`
    ------
    `{refine}`
    '''

    refine_prompt = PromptTemplate(
        template=refine_template,
        input_variables=['existing_answer', 'text', 'refine']
    )

    chain = load_summarize_chain(
        llm = llm,
        chain_type = 'refine',
        question_prompt = initial_prompt,
        refine_prompt = refine_prompt,
        return_intermediate_steps = False
    )

    output_summary = chain.invoke({'input_documents': chunks, 'refine': final_prompt})
    return output_summary['output_text']

def lc_agent(llm, question, agent):
    from langchain.utilities import WikipediaAPIWrapper
    from langchain.agents import initialize_agent, Tool

    wikipedia = WikipediaAPIWrapper()

    tools = [
        Tool(
            name = 'Intermediate Answer',
            func = wikipedia.run,
            description = 'Useful for when you need to get information from Wikipedia about a document.'
            # not mandatory but useful
        )
    ]  # Ways an agent can interact with the outside world.

    agent_executer = initialize_agent(tools,
                                      llm,
                                      agent = agent,
                                      verbose = False,
                                      handle_parsing_errors = True)

    output = agent_executer.invoke(question)
    return output['output']

if __name__ == '__main__':
    st.set_page_config(
        page_title = 'Document Summariser',
        page_icon = '../assets/catworking.png',
        layout = 'wide'
    )

    # Initialising document history
    if "document" not in st.session_state:
        st.session_state.document = ''

    with (st.sidebar):
        # Get the API Key
        openai_key = st.text_input(label='OpenAI API Key', type='password')
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
        st.write()

        # Choosing the Model Type
        model = st.segmented_control(label = 'OpenAI Model', options = ['gpt-4o','gpt-3.5-turbo','gpt-4o-mini'], default = 'gpt-3.5-turbo', selection_mode = 'single' , help = 'This model will be used to summarise you document or text.')
        if not model:
            model = 'gpt-3.5-turbo'
            st.warning('No model has been selected, so gpt-3.5-turbo will be used by default')
        llm = ChatOpenAI(model_name = model, temperature = 0)
        st.divider()

        # Loading Data
        data_type = st.radio(label = 'Data Format', options = ['Text','Document'], on_change = clear_document, help = 'The format of the data you wish to summarise. Document types that are accepted include PDF, docs, and txt files. Text refers to plain text that is typed or pasted into the text input box.')
        if data_type == 'Text':
            data = st.text_area(label = 'Enter text', placeholder = 'Enter your text...', height = 200, label_visibility = 'collapsed')
        elif data_type == 'Document':
            data = st.file_uploader('Upload a file:', on_change = clear_document, type=['pdf', 'txt', 'docx'])

        convert_doc = st.button('Load Data')

        if data and convert_doc:
            with st.spinner('Uploading Data...'):
                if data_type == 'Document':
                    bytes_data = data.read()
                    file_name = os.path.join('./', data.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)

                    st.session_state.document = load_document(file_name)
                elif data_type == 'Text':
                    from langchain.docstore.document import Document
                    st.session_state.document = [Document(page_content=data)]
                st.success('Data uploaded')
                if 'document' in st.session_state and st.session_state.document:
                    all_text = ''.join(doc.page_content for doc in st.session_state.document)
                    st.session_state.all_text = all_text

    ## Main Page
    st.subheader('Document Summarisation')

    tab1, tab2, tab3 = st.tabs(['**Summarise a short document**','**Summarise a large document**','**Summarise information from Wikipedia**'])
    with tab1:
        if st.session_state.document:
            short_options = {
                'Provide a concise summary': 'none',
                'Summarise in a different language': 'lang',
                'Summarise using specific instructions': 'inst'
            }
            short_select = st.radio(label='**How would you like to summarise the document?**', options=short_options.keys(), horizontal=True)
            if short_options[short_select] == 'none':
                with st.form('s1', enter_to_submit = True):
                    st.write('A short and concise summary will be generated.')
                    s1 = st.form_submit_button('Generate Summary')

                if s1:
                    with st.spinner('Generating Summary...'):
                        st.write_stream(stream_data(short_doc_no_instruction(llm, st.session_state.document)))
            if short_options[short_select] == 'lang':
                with st.form('s2', enter_to_submit = True):
                    lang = st.text_input(label = 'What language should the summary be translated to?')

                    s2 = st.form_submit_button('Generate Summary')
                if lang and s2:
                    with st.spinner('Generating Summary...'):
                        st.write_stream(stream_data(short_doc_lang(llm, st.session_state.document,lang)))
                elif s2 and lang == '':
                    st.warning('Please enter a language')

            if short_options[short_select] == 'inst':
                with st.form('s3', enter_to_submit = True):
                    inst = st.text_area(label = 'Specify the instructions that must be followed when summarising your document or text', placeholder = 'Write an in depth summary of the following that covers the key points. Add a title to the summary. Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.')

                    s3 = st.form_submit_button('Generate Summary')
                if inst and s3:
                    with st.spinner("Generating summary..."):
                        st.write(short_doc_instructions(llm,st.session_state.document,inst))
                elif inst == '' and s3:
                    st.warning('Please enter an instruction')

    with tab2:
        if st.session_state.document:
            col1, col2 = st.columns([0.65, 0.35])
            instr_type = {
                'Provide a concise summary': 'none',
                'Summarise using specific instructions': 'instr'
            }
            instruction_long = col1.radio('**How would you like to summarise the document?**', options = instr_type.keys(), horizontal = True)
            chain_types = {
                'Map Reduce': 'map_reduce',
                'Refine': 'refine'
            }
            chain = col2.radio(label = '**Select a chain type**', options = chain_types.keys(), horizontal = True)

            total_characters = len(st.session_state.all_text)

            # Chain Type = Map Reduce and No Prompt
            if instr_type[instruction_long] == 'none' and chain_types[chain] == 'map_reduce':
                with st.form('l1'):
                    colL11, colL12 = st.columns([0.5, 0.5])
                    num_chunks = colL11.number_input('Number of characters per chunk',
                                                     step = 1, value = int((total_characters/2)),
                                                     min_value = 0, max_value = total_characters,
                                                     help = 'The default value is half the number of character in a chunk is half the characters in the document, resulting in 2 chunks.')
                    overlap = colL12.number_input('Number of character overlap per chunk',
                                                  step = 1, value = int((total_characters*0.01)),
                                                  min_value = 0, max_value = total_characters,
                                                  help = 'The default number of character overlap per chunk is 1% of the characters in the document.')

                    l1 = st.form_submit_button('Generate Summary')
                if l1:
                    with st.spinner('Generating Summary...'):
                        st.write_stream(stream_data(long_doc_map_no_instructions(llm, st.session_state.all_text, num_chunks, overlap)))

            # Chain Type = Refine and No Prompt
            if instr_type[instruction_long] == 'none' and chain_types[chain] == 'refine':
                with st.form('l2'):
                    colL21, colL22 = st.columns([0.5, 0.5])
                    num_chunks = colL21.number_input('Number of characters per chunk',
                                                     step=1, value=int((total_characters / 2)),
                                                     min_value=0, max_value=total_characters,
                                                     help='The default value is half the number of character in a chunk is half the characters in the document, resulting in 2 chunks.')
                    overlap = colL22.number_input('Number of character overlap per chunk',
                                                  step=1, value=int((total_characters * 0.01)),
                                                  min_value=0, max_value=total_characters,
                                                  help='The default number of character overlap per chunk is 1% of the characters in the document.')

                    l2 = st.form_submit_button('Generate Summary')
                if l2:
                    with st.spinner('Generating Summary...'):
                        st.write_stream(
                            stream_data(long_doc_refine_no_instructions(llm, st.session_state.all_text, num_chunks, overlap)))

            # Chain Type = Map Reduce and Prompt
            if instr_type[instruction_long] == 'instr' and chain_types[chain] == 'map_reduce':
                with st.form('l3'):
                    colL31, colL32 = st.columns([0.5, 0.5])
                    num_chunks = colL31.number_input('Number of characters per chunk',
                                                    step=1, value=int((total_characters / 2)),
                                                    min_value=0, max_value=total_characters,
                                                    help='The default value is half the number of character in a chunk is half the characters in the document, resulting in 2 chunks.')
                    overlap = colL32.number_input('Number of character overlap per chunk',
                                                    step=1, value=int((total_characters * 0.01)),
                                                    min_value=0, max_value=total_characters,
                                                    help='The default number of character overlap per chunk is 1% of the characters in the document.')
                    prompt = st.text_input(label = 'Final prompt')

                    l3 = st.form_submit_button('Generate Summary')
                if l3 and prompt:
                    with st.spinner('Generating Summary...'):
                        st.write(long_doc_map_prompt(llm, st.session_state.all_text, prompt, num_chunks, overlap))
                elif l3 and not prompt:
                    st.warning('Enter a final prompt')

            # Chain type = Refine and Prompt
            if instr_type[instruction_long] == 'instr' and chain_types[chain] == 'refine':
                with st.form('l4'):
                    colL41, colL42 = st.columns([0.5, 0.5])
                    num_chunks = colL41.number_input('Number of characters per chunk',
                                                     step=1, value=int((total_characters / 2)),
                                                     min_value=0, max_value=total_characters,
                                                     help='The default value is half the number of character in a chunk is half the characters in the document, resulting in 2 chunks.')
                    overlap = colL42.number_input('Number of character overlap per chunk',
                                                  step=1, value=int((total_characters * 0.01)),
                                                  min_value=0, max_value=total_characters,
                                                  help='The default number of character overlap per chunk is 1% of the characters in the document.')
                    prompt = st.text_input(label='Final prompt')

                    l4 = st.form_submit_button('Generate Summary')
                if l4 and prompt:
                    with st.spinner('Generating Summary...'):
                        st.write(long_doc_refine_prompt(llm, st.session_state.all_text, prompt, num_chunks, overlap))
                if l4 and not prompt:
                    st.warning('Enter a final prompt')

    with tab3:
        st.write('Based on your question and the reasoning method you select, an agent will create a summary by summarising the relevant information from Wikipedia using the reasoning method selected.')
        with st.form('agent', enter_to_submit = True):
            question = st.text_input(label = 'Enter your question')
            agent_options = {
                'A zero shot agent that does a reasoning step before acting.':'zero-shot-react-description',
                'An agent that breaks down a complex question into a series of simpler questions.': 'self-ask-with-search'
            }
            agent = st.radio(label = 'What kind of reasoning method should the agent use:', options = agent_options.keys())

            w1 = st.form_submit_button('Generate Summary')
        if question and agent and w1:
            with st.spinner("Generating summary..."):
                st.write_stream(stream_data(lc_agent(llm,question,agent_options[agent])))
        elif w1 and not question:
            st.warning('Please enter your question')
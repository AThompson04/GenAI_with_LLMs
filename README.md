# Generative AI with Large Language Models (LLM)

## Overview

> Generative AI is a subset of AI, where generative models are used to create content such as text, images etc. This project will focus on generating text content.

This project consists of three generative AI models:
1. **Chatbot:**<br/>LLM based chatbots generate contextually relevant responses to the end user's queries while considering the entire interaction between chatbot and end user, and personalising its interaction to meet the end user's goal.  
2. **Retrieval-Augmented Generation (RAG) Chatbot:**<br/>A Framework that uses LLMs and private data to provide the end user with more accurate, up-to-date and contextually relevant answers. This is done through retrieval and generation. Relevant information on the end user's query is found in the private documents and sent to the LLM. The model then generates a response using the end user's query, the private information received and its own internal knowledge.
3. **Summariser:**<br/>With use of LLMs concise or specifically structured summaries are generated from longer text documents. The summaries can be generated using a variety of techniques which are showcased in this project.<br/><br/>

## Models
This section will highlight how each of the models work and have been structured.
### Chatbot


Using Langchain_openai's ChatOpenAi
Model got-3.5-turbo by default but this could be changed. Not that changes in model will change the price of each request. Temperature = 0.5 (measure between 0 and 1 which determines the 'creativeness' of the response. 0 indicates that the responses will be straightforward and predictable aka you will always get the same response to the same prompt. 1 indicating that the responses can vary widely to prompt. An extension to the current model would be to let the end-user change the model temperature. Warning that if the temperature is high the chatbot can start 'hallucinating' which means the model could produce misleading or incorrect results. Add how temperature works? LLMs work by selecting the next best word given the prompt so low temperature = high probability next words and high temperatures = words with lower probability.) Because there is no specific purpose for this chatbot I selected 0.5 so that you could perform some more creative tasks as well.
System prompt - how you would like the model to act. If none is given the model is asked to act as a helpful assistant. This could be used to ask the model to respond in a specific role I.e. HR Manager, or to respond in a different language etc. Please not that this cannot be changed after your initial prompt to the chatbot however you can specify in your prompts to respond to the particular prompt differently (or from now on... test).
The user is prompted to give a prompt.
This is added to the chatbot 'memory', the session state.
The entire conversation and the system prompt, the session state, are set to the model select and the response is returned and displayed to the end user. This response is saved in the memory, session state. Add the reason for storing and sending the memory is so that the chatbot can give the most relevant answers to the question based on previous questions and so you can ask follow up questions or reference other questions.

### RAG Chatbot

User adds their document, if they would like to specify specifies the chuck size (the number of characters in a chunk - the more letters in a chuck the more information is sent to the LLM which could result in a more accurate answer. If the chuck size is too small you may lose information. EXPLAIN more) and the number of similar items that must be retrieved and sent to the LLM (k).
The model created a Chroma storage directory where the embedded data will be stored.
The data is split into chucks.
The chucks are used to create embeddings (Because computers cannot understand text, the text needs to be transformed into a form which can be understood which is numerical vectors. Embeddings are real world objects like text, images or videos that are converted into a form that computers can process - a vector. By creating embeddings similarity searches can be preformed - finding pieces of text that are similar to each other by finding vectors that are similar to each other. Add picture with [reference](https://towardsdatascience.com/mastering-customer-segmentation-with-llm-3d9008235f41#3a33)) and stored in the Embeddings done using langchain_openai's OpenAIEmbeddings with the model 'text-embeddind-3-small'. Chroma Storage directory and as a vector.
When a new document is uploaded, any information from the previous document will be discarded. This is done by clearing the Chroma Storage Directory.
Once the document has been uploaded the end-user can prompt the chatbot. The model will find the k most similar/relevant chucks of information according to the prompt from the document. These chucks will be sent to the LLM along with the users prompt and all the chat history - reason (so that the chatbot can give the most relevant answers to the question based on previous questions and so you can ask follow up questions or reference other questions). The LLM will generate an answer which will be displayed in the chat. Both the prompt and the answer will be saved to the session state, 'memory'. The LLM has instructions (a system prompt) to use its own knowledge outside the context if the context/chucks do not contain information about the question.
The LLM in use is ChatOpenAI model = get-3.5-turbo by default but can select another (check). Temperature = 0

### Summariser

The data summariser can either summarise text typed into the console or summarise a document. The three document types that it accepts is pdf, txt or docx. The document or text will be loaded and stored to the models memory. There are three options:
1. Provide a summary for a short document
2. Provide a summary for a long document
3. Provide a summary based on information on Wikipedia
Each of the options mentioned above use *langchain_openai*'s *ChatOpenAI* function. The OpenAI model used by default is 'gpt-3.5-turbo' but the end-user can change this to 'gpt-4o' or 'gpt-4o-mini'.

**More details about each summarisation option:**
1. **Summarise a Short Document:**<br/>When summarising a short document you are given three summarisation options - to provide a concise summary, provide the concise summary in a different language or to summarise the document using specific instructions i.e. include an introduction and conclusion, and summarise all main points using bullet points. For each summarisation option prompts are used to tell the LLM how to summarise the data, this was done using *langchain_core*'s *PromptTemplate* function.<br/><br/>When providing a concise summary the LLM prompt is as follows "You are an expert copywriter with expertise in summarising documents. Write a summary of the following text.".<br/><br/>When converting to a different language the LLM prompt is to "Write a summary of the following text and to translate the summary to *the language given by the end user*".<br/><br/>When summarising using specific instructions the prompt is entirely the instruction from the end-user.<br/><br/>The information for each option is passed to the LLM using the *stuff* chain approach, i.e. the entire document is summarised in a single prompt. See the visual explaination in Figure 2 below by [Rahul](https://ogre51.medium.com/types-of-chains-in-langchain-823c8878c2e9). Once the summary has been generated by the LLM, it is displayed by the model.<br/><figure><img src='/assets/stuff.webp' style="width: 65%; height: auto;"><br/><figcaption>*Figure 2: Stuff Chain Type Visualisation*</figcaption></figure><br/><br/>
2. **Summarise a Long Document:**<br/>When summarising a longer document you are given two sets of options - the first refers to the instructions sent to the LLM on how to summarise the document, and the second refers to how the information is sent the the LLM (chain type). For the first option the LLM is asked to provide a concise summary or to summarise the document using specific instructions. The second option sends the information to the LLM using either the Map Reduce or Refine chain type.<br/><br/>The Map Reduce chain type chunks the document(s) and instructs the LLM to summarise the chunks of the document separately and then to provide a final summary by summarising the summarises of each chunk. See the visual explaination in Figure 3 below by [Rahul](https://ogre51.medium.com/types-of-chains-in-langchain-823c8878c2e9).<br/><figure><img src='/assets/map_refine.webp' style="width: 65%; height: auto;"><br/><figcaption>*Figure 3: Map Reduce Chain Type Visualisation*</figcaption></figure><br/><br/>The Refine chain type chunks the document(s) and instructs the LLM to create a summary of the first chunk, the to create another summary using the next chunk and the previous summary. The final summary is produced by summarising the last chunk of the document and the previous summary. See the visual explaination in Figure 4 below by [Rahul](https://ogre51.medium.com/types-of-chains-in-langchain-823c8878c2e9). <br/><figure><img src='/assets/refine.webp' style="width: 65%; height: auto;"><br/><figcaption>*Figure 4: Refine Chain Type Visualisation*</figcaption></figure><br/><br/>If the summarisiation instructions are to provide a concise summary, then the LLM will create the final summary as it created the previous summaries. Otherwise, if you like to LLM to summarise using sepcific instriuctions then the instruction will be added to a final prompt, using *langchain_core*'s *PromptTemplate* function. The inital prompt for both the map reduce and refine approach is to write a concise summary. This will be used to generate summaries for all the chunks for the map reduce approach, and all the summaries except the final summary for the refine chain type. The final prompt with the end-user's specific instructions will be used when summarising all the chunk summaries in the Map Reduce approach and to generate the final summary using the last chuck and previous summary for the Refine approach. Once the final summary has been generated by the LLM, it is displayed by the model.<br/><br/>
3. **Summarise Wikipedia Information:**<br/>When summarising information from Wikipedia the end-user needs to provide a question or what they would like information on, and select the reasoning method that they would like the agent to use when summarising the data. The reasoning methods available for this model are 'zero-shot-react-description' and 'self-ask-with-search'.<br/><br/>'zero-shot-react-description' refers to the reasoning method where the agent does not ask additional questions but rather generates a response based on the information that it has.<br/><br/>'self-ask-with-search' refers to the reasoning method where the agent will ask itself additional questions about the end-users topic and based on its findings to these questions generate a more informed response.<br/><br/>Wikipedia access is done through *langchain*'s *WikipediaAPIWrapper* function, and the agent is set up using *langchain*'s *initialize_agent* function. Once the agent has an answer to the end-user's prompt it will be displayed by the model.<br/><br/>

## Setting Up Your Environment / Prerequisites

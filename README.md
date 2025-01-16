# Generative AI with Large Language Models (LLM)

## Overview

> Generative AI is a subset of AI, where generative models are used to create content such as text, images etc. This project will focus on generating text content.

This project consists of three generative AI models:
1. **Chatbot:**<br/>LLM based chatbots generate contextually relevant responses to the end user's queries while considering the entire interaction between chatbot and end user, and personalising its interaction to meet the end user's goal.  
2. **Retrieval-Augmented Generation (RAG) Chatbot:**<br/>A Framework that uses LLMs and private data to provide the end user with more accurate, up-to-date and contextually relevant answers. This is done through retrieval and generation. Relevant information on the end user's query is found in the private documents and sent to the LLM. The model then generates a response using the end user's query, the private information received and its own internal knowledge.
3. **Summariser:**<br/>With use of LLMs concise or specifically structured summaries are generated from longer text documents. The summaries can be generated using a variety of techniques which are showcased in this project.

## Models
This section will highlight how each of the models have been built.
#### Chatbot


Using Langchain_openai's ChatOpenAi
Model got-3.5-turbo by default but this could be changed. Not that changes in model will change the price of each request. Temperature = 0.5 (measure between 0 and 1 which determines the 'creativeness' of the response. 0 indicates that the responses will be straightforward and predictable aka you will always get the same response to the same prompt. 1 indicating that the responses can vary widely to prompt. An extension to the current model would be to let the end-user change the model temperature. Warning that if the temperature is high the chatbot can start 'hallucinating' which means the model could produce misleading or incorrect results. Add how temperature works? LLMs work by selecting the next best word given the prompt so low temperature = high probability next words and high temperatures = words with lower probability.) Because there is no specific purpose for this chatbot I selected 0.5 so that you could perform some more creative tasks as well.
System prompt - how you would like the model to act. If none is given the model is asked to act as a helpful assistant. This could be used to ask the model to respond in a specific role I.e. HR Manager, or to respond in a different language etc. Please not that this cannot be changed after your initial prompt to the chatbot however you can specify in your prompts to respond to the particular prompt differently (or from now on... test).
The user is prompted to give a prompt.
This is added to the chatbot 'memory', the session state.
The entire conversation and the system prompt, the session state, are set to the model select and the response is returned and displayed to the end user. This response is saved in the memory, session state. Add the reason for storing and sending the memory is so that the chatbot can give the most relevant answers to the question based on previous questions and so you can ask follow up questions or reference other questions.

#### RAG Chatbot

User adds their document, if they would like to specify specifies the chuck size (the number of characters in a chunk - the more letters in a chuck the more information is sent to the LLM which could result in a more accurate answer. If the chuck size is too small you may lose information. EXPLAIN more) and the number of similar items that must be retrieved and sent to the LLM (k).
The model created a Chroma storage directory where the embedded data will be stored.
The data is split into chucks.
The chucks are used to create embeddings (Because computers cannot understand text, the text needs to be transformed into a form which can be understood which is numerical vectors. Embeddings are real world objects like text, images or videos that are converted into a form that computers can process - a vector. By creating embeddings similarity searches can be preformed - finding pieces of text that are similar to each other by finding vectors that are similar to each other. Add picture with [reference](https://towardsdatascience.com/mastering-customer-segmentation-with-llm-3d9008235f41#3a33)) and stored in the Chroma Storage directory and as a vector.
When a new document is uploaded, any information from the previous document will be discarded. This is done by clearing the Chroma Storage Directory.
Once the document has been uploaded the end-user can prompt the chatbot. The model will find the k most similar/relevant chucks of information according to the prompt from the document. These chucks will be sent to the LLM along with the users prompt and all the chat history - reason (so that the chatbot can give the most relevant answers to the question based on previous questions and so you can ask follow up questions or reference other questions). The LLM will generate an answer which will be displayed in the chat. Both the prompt and the answer will be saved to the session state, 'memory'. The LLM has instructions (a system prompt) to use its own knowledge outside the context if the context/chucks do not contain information about the question.
The LLM in use is ChatOpenAI model = get-3.5-turbo by default but can select another (check). Temperature = 0

#### Summariser

## Setting Up Your Environment / Prerequisites
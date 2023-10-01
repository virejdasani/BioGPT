import torch
from typing import Any, Dict, List
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain import HuggingFacePipeline
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
from streamlit_chat import message
import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from backend.biogpt import biogpt_response


os.environ["OPENAI_API_KEY"] = "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"

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")




def main():
    # Initialize the retriever
    retriever = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Start the chat
    while True:
        query = st.text_input("Ask a question:")

        # Get the response from the LLM
        response, chat_history = biogpt_response(query, chat_history)

        # Display the response
        st.write(response)

        # Update the chat history
        retriever.update_memory(chat_history)


if __name__ == "__main__":
    main()

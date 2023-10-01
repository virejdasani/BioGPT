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

os.environ["OPENAI_API_KEY"] = "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"

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

#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
def biogpt_response(query: str, chat_history: List[Dict[str, Any]] = []):
    inputs = tokenizer(query, return_tensors="pt")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    set_seed(42)
     
    with torch.no_grad():
        beam_output = model.generate(
            **inputs, min_length=100, max_length=1024, num_beams=5, early_stopping=True
        )
       
    response = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    
    # Add the current query and response to the chat history
    chat_history.append({"prompt": query, "response": response})

    return response, chat_history



memory = ConversationBufferMemory(memory_key="chat_history")

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

set_seed(42)

llm_chain = LLMChain(llm=model, prompt="COVID-19 is", verbose=True, memory=memory)

llm_chain.predict(human_input="a global pandemic")

print(llm_chain.predict(human_input="caused by"))

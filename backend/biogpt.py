import torch
from typing import Any, Dict, List
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain import HuggingFacePipeline
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
from streamlit_chat import message
import os


#os.environ["OPENAI_API_KEY"] = "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"

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

#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
def biogpt_response(query: str, chat_history: List[Dict[str, Any]] = []):
    inputs = tokenizer(query, return_tensors="pt")

    template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=biogpt_llm,
        verbose=True,
        memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
    )
      llm_chain = LLMChain(
        prompt = long_prompt,
        llm = biogpt_llm
     )
    conversation.predict(input=llm_chain.run("What is Covid 19?"))

    conversation.predict(input=llm_chain.run("What are its symptoms"))  
  

    set_seed(42)
     
    with torch.no_grad():
        beam_output = model.generate(
            **inputs, min_length=100, max_length=1024, num_beams=5, early_stopping=True
        )
       
    response = tokenizer.decode(beam_output[0], skip_special_tokens=True)
    chat_history.append({"prompt": query, "response": response})

    return response, chat_history





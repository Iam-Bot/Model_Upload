# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python (Local)
#     language: python
#     name: base
# ---

# +
import logging
import json
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate  
from Config import get_vertex_ai_llm
import warnings

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_prompts():
    with open('Prompts.json','rb') as file:
        return json.load(file)
prompts=load_prompts()

class DeveloperChatbot:
    """
    A class for a chatbot using LangChain and VertexAI.
    """
    
    def __init__(self, model, template):
        """
        Initialize the chatbot with given parameters.

        Args:
            model: The language model to use for generating responses.
            template: The template for generating prompts.
        """
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"], template=template
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.llm = model
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)
        logging.info("Chatbot initialized successfully.")

    def reset_memory(self):
        """
        Reset the memory for the conversation history.
        """
        self.memory.clear()
        logging.info("Chatbot memory cleared.")

    def handle_conversation(self, user_input):
        """
        Handle the conversation and generate responses.

        Args:
            user_input (str): The input from the user.

        Returns:
            str: The response from the chatbot.
        """
        if user_input.lower() == "quit":
            logging.info("Conversation ended by user.")
            return "Conversation ended."
        
        response = self.llm_chain.run(user_input)
        return response

# Define the chatbot template
chat_template = prompts['code_assistant']['chat_template']

# Initialize the chatbot instance
chatbot = DeveloperChatbot(
    model=get_vertex_ai_llm('llm'),
    template=chat_template
)



chatbot.reset_memory()
user_input = """Convert the following code to Java language.

Code:

//XIRU8000 JOB (CRIS00M0),'RUN OF U8000',MSGCLASS=T,
//         NOTIFY=&SYSUID,MSGLEVEL=(1,1)
//*MAIN       CLASS=DB2BAT,SYSTEM=SY6D
//OUTDEF  OUTPUT  NAME=&SYSUID,
// ADDRESS=('5325 ZUNI ST.','DENVER      CO','80221'),
// JESDS=ALL,DEFAULT=Y
//*
//PROCS JCLLIB ORDER=(DBS1.#JB.COMP.PROC,
//      XIR.#SU.DUEDATE1.PROC,
//      XIR.#SU.PROD.PROC,XIR.#SU.PROC)
//RELEASE SET ENV=E,
//            REG=CR,
//            SITE=C,
//            GDEFIL=JSANCHE.APA8,
//            CYCLE=Z1,
//            RENV=SU,
//            STEPLIB=DUEDATE,
//            XIRREL=NEWPROD,
//            RLSE=PROD              <== MODIFY FOR TEXT LOADS
//*
//*****************************************************************
//* XIRU8000 - NEW PROGRAM FOR CREATING GUIDE FILE
//*****************************************************************
//XIRU8000 EXEC XIRU800P,
//         SPLR=SPLRCENT,
//         BLGGRP=BLGGRPCT,
//         RLSE=&RLSE,
//         GUIDECTL=&GDEFIL,
//         DB2CARD=DB2U800&ENV,            ENV*
//         U800001O=XIR.#SU.&REG.U&ENV..U800001O.&CYCLE(+1),
//         U800002O=XIR.#SU.&REG.U&ENV..U800002O.&CYCLE(+1),
//         U800003O=XIR.#SU.&REG.U&ENV..U800003O.&CYCLE(+1),
//         COND=(4,LT)
//*
//Z900001O DD SYSOUT=*
/*

"""
response = chatbot.handle_conversation(user_input)
print("--------------------------------------------------------")

print(response)
print("--------------------------------------------------------")
# if __name__ == "__main__":
#     chatbot.reset_memory()
#     print("Chatbot is ready. Type 'quit' to end the conversation.")
#     while True:
#         user_input = input("Developer: ")
#         if user_input == "quit":
#             break
#         else:
#             response = chatbot.handle_conversation(user_input)
#             print(f"Chatbot: {response}")
#             print("-------------------------------------------------------------")


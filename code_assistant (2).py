"""
File Name: developer_chatbot.py
Author: Gopesh, Mayank (Version 1), Denesh A (Version 2)
Purpose: This script defines the DeveloperChatbot class to handle WebSocket communication with a chatbot using LangChain and VertexAI. 
         The Code Assistant module acts as a conversational AI assistant for developers, helping with code optimization, translation, and test case generation.
Date: 2024-05-20

Examples:
    - Initialize the DeveloperChatbot:
        chatbot = DeveloperChatbot(model, template)
    - Handle a conversation via WebSocket:
        await chatbot.handle_conversation(websocket)
Realtime example:
    - Translate the python code to Fortran
        var nterms = int(input("How many terms? "))
        var n1 = 0
        var n2 = 1
        var count = 0

        if nterms <= 0:
            print("Please enter a positive integer")
        elif nterms == 1:
            print("Fibonacci sequence upto", nterms, ":")
            print(n1)
        else:
            print("Fibonacci sequence:")
            while count < nterms:
                print(n1)
                var nth = n1 + n2
                n1 = n2
                n2 = nth
                count += 1
    
    - ```fortran
        PROGRAM fibonacci
        IMPLICIT NONE
        INTEGER :: nterms, n1, n2, count, nth

        WRITE(*,*) "How many terms? "
        READ(*,*) nterms

        IF (nterms <= 0) THEN
            WRITE(*,*) "Please enter a positive integer"
        ELSEIF (nterms == 1) THEN
            WRITE(*,*) "Fibonacci sequence upto", nterms, ":"
            WRITE(*,*) n1
        ELSE
            WRITE(*,*) "Fibonacci sequence:"
            n1 = 0
            n2 = 1
            count = 0
            DO WHILE (count < nterms)
            WRITE(*,*) n1
            nth = n1 + n2
            n1 = n2
            n2 = nth
            count = count + 1
            END DO
        END IF
        END PROGRAM fibonacci
        ```

"""

import os
import sys
import logging
from fastapi import WebSocket, WebSocketDisconnect
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from Prompts import prompts  # Import prompts from the prompts module

# Append the parent directory of Generator to sys.path to access all sibling folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Config import setup_environment, get_vertex_ai_llm

# Set up logging
logging.basicConfig(level=logging.INFO)

class DeveloperChatbot:
    """
    A class that handles WebSocket communication with a chatbot using LangChain and VertexAI.
    """
    
    def __init__(self, model, template):
        """
        Initialize the chatbot with given parameters.

        Args:
            model: The language model to use for generating responses.
            template: The template for generating prompts.

        Example:
            chatbot = DeveloperChatbot(model, template)
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

        Example:
            chatbot.reset_memory()
        """
        self.memory.clear()
        logging.info("Chatbot memory cleared.")

    async def handle_conversation(self, websocket: WebSocket):
        """
        Manage the conversation via WebSocket.

        Args:
            websocket (WebSocket): The WebSocket connection for the conversation.

        Example:
            await chatbot.handle_conversation(websocket)
        """
        await websocket.accept()
        self.reset_memory()
        try:
            while True:
                try:
                    usr_ip = await websocket.receive_text()
                    if usr_ip == "quit":
                        await websocket.close()
                        break
                    response = self.llm_chain.run(usr_ip)
                    await websocket.send_text(response)
                except WebSocketDisconnect:
                    logging.warning("WebSocket connection closed unexpectedly.")
                    break
                except Exception as e:
                    logging.error(f"Error processing input: {e}")
                    await websocket.send_text("Error processing your request")
        finally:
            await websocket.close()
            logging.info("WebSocket connection closed.")

# Load environment variables for credentials
setup_environment()

# Define the chatbot template
chat_template = prompts['code_assistant']['chat_template']

# Initialize the chatbot instance
chatbot = DeveloperChatbot(
    model=get_vertex_ai_llm('llm'),
    template=chat_template
)

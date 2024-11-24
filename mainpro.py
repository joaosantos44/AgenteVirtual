from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Pinecone as PipeconeLangchain
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents import (create_react_agent, AgentExecutor)
from langchain.chains.retrieval import create_retrieval_chain
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import AgentExecutor
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import Tool
from streamlit_chat import message
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import Any, Dict
from langchain import hub
import streamlit as st
import datetime
import os

load_dotenv()



def save_history(question, answer):
    with open("history.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}: {question}->{answer}\n")

def load_history():
    if os.path.exists("history.txt"):
        with open("history.txt", "r") as f:
            return f.readlines()
    return[]

def main():
    st.set_page_config(page_title="Agente de Python",
                       page_icon="👾",
                       layout="wide")
    st.title("👾 Agente de Python 👾")
    st.markdown(
        """
        <style>
        .stApp{background-color: black;}
        .title{color=#ff4b4b;}
        .button {background-color: #ff4b4b; color: white; border-radius: 5px;} 
        .input{border: 1px solid ##ff4b4b; border-radius: 5px;}
        </style>
        """,
        unsafe_allow_html = True
    )

    instrucciones = """
    - Siempre usa la herrameinta, incluso si sabes la respuesta.
    - Debes usar codigo de Python para responder.
    - Eres un agente que puede escribir codigo.
    - Solo responde la pregunta escribiendo codigo, incluso si sabes la respuesta.
    - Si no sabes la respuesta escribe "No se la respuesta".
    """

    st.markdown(instrucciones)

    base_prompt =hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instrucciones=instrucciones)
    st.write("Cargando...")

    tool = [PythonREPLTool()]
    llm = ChatOpenAI(model = "gpt-4o", temperature = 0)
    agent = create_react_agent(
        llm=llm,
        tools=tool,
        prompt=prompt
    )


    agent_executor = AgentExecutor(
        agent=agent,
        tools=tool,
        path=["episode_info.csv", "classes.csv", "elden_ring_weapon.csv", "equipment.csv", "monsters.csv", "Pokemon.csv", "races.csv", "spells.CSV"],
        verbose=True,
        handle_parsing_errors=True,
    )

    st.markdown("### Listado: ")

    ejemplos = [
        "calcula la suma de 2 y 3",
        "genera una lista del 1 al 10",
        "Crea una funcion que calcule el factorial de un numero",
        "Crea un juego basico de snake con la libreria pygame"
    ]

    example = st.selectbox("Selecciona una opcion:", ejemplos)

    if st.button("Ejecutar opcion"):
        user_input = example
        try:
            respuesta = agent_executor.invoke(input={"input": user_input, "instructions": instrucciones, "agent_scratchpad": ""})
            st.markdown("### Resultado del agente:")
            st.code(respuesta["output"], language="python")
            save_history(user_input, respuesta["output"])
        except  ValueError as e:
            st.error(f"Error en el agente: {str(e)}")

    st.markdown("### Preguntas: ")

    contenido = """
        El contenido del que se le puede preguntar al agente es sobre Pokemon, Elden Ring y D&D.
        """

    st.markdown(contenido)

    prompt = st.text_input("Prompt", placeholder="Enter Your prompt here")


    instru2 = """You are an agent designed to write and execute Python code to answer questions.
        You have access to a python REPL, which you can use to execute python code.
        You have qrcode package installed
        If you get an error, debug your code and try again.
        Only use the output of your code to answer the question.
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """
    if st.button("Ejecutar pregunta"):
        user_input2 = prompt
        try:
            respuesta = agent_executor.invoke(input={"input": user_input2, "instructions": instru2, "agent_scratchpad": ""})
            st.markdown("### Resultado del agente:")
            st.code(respuesta["output"], language="python")
            save_history(user_input2, respuesta["output"])
        except  ValueError as e:
            st.error(f"Error en el agente: {str(e)}")

if __name__ == "__main__":
    main()

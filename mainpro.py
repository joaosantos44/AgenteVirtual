import streamlit as st
from langchain_experimental.tools import PythonREPLTool
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
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
                       page_icon="⚔️",
                       layout="wide")
    st.title("⚔️ Agente de Python")
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
    st.write("Prompt cargando...")

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
        verbose=True,
        handle_parsing_errors=True,
    )

    st.markdown("### Ejemplos: ")

    ejemplos = [
        "calcula la suma de 2 y 3",
        "genera una lista del 1 al 10",
        "Crea una funcion que calcule el factorial de un numero",
        "Crea un juego basico de snake con la libreria pygame"
    ]

    example = st.selectbox("Selecciona una opcion:", ejemplos)

    if st.button("Ejecutar ejemplo"):
        user_input = example
        try:
            respuesta = agent_executor.invoke(input={"input": user_input, "instructions": instrucciones, "agent_scratchpad": ""})
            st.markdown("### Resultado del agente:")
            st.code(respuesta["output"], language="python")
            save_history(user_input, respuesta["output"])
        except  ValueError as e:
            st.error(f"Error en el agente: {str(e)}")


if __name__ == "__main__":
    main()
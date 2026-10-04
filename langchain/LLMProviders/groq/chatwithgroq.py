from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
#groq needs proxy
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1080"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:1080"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"
# if "GROQ_API_KEY" not in os.environ:
#     os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

def ollamaExample():
    # Initialize the model with your custom name
    llm = ChatOllama(
        model="kevinllama3.2:3b",  # Your custom model name
        temperature=0.7,
    )

    # Basic invocation
    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "build RAG agent."),
    ]

    response = llm.invoke(messages)
    print("Ollama LLM:")
    print(response.content)

def groqExample():
    # Initialize the model with your custom name
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        # api_key="",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=1
    )

    # Basic invocation
    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "hi"),
    ]

    response = llm.invoke(messages)
    print("Grop LLM:")
    print(response.content)

#========================================================
groqExample()
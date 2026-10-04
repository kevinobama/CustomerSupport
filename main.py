from fastapi import FastAPI
from langchain.agents import create_agent

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Bill Gates"}


@app.get("/openai/v1/chat/completions")
async def completions():
    agent = create_agent(
        model="kevinllama3.2:3b",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
    )
    return result


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


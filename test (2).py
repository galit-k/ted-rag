import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. Load the keys from .env
load_dotenv()

# 2. Configure the Model using LLMod.ai details
llm = ChatOpenAI(
    api_key=os.getenv("LLMOD_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    model="gpt-5-mini"  # Model specified in assignment [cite: 311]
)

# 3. Test it
try:
    print("Sending test message...")
    response = llm.invoke("Hello! Are you ready for the TED Talk assignment?")
    print("\nSUCCESS! Model replied:")
    print(response.content)
except Exception as e:
    print("\nERROR: Could not connect.")
    print(e)
import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.llms import Anthropic
from langchain_anthropic import ChatAnthropic

load_dotenv(find_dotenv())

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620",
                      temperature= 0.5,
                      max_tokens=1024,
                      timeout=None,
                      max_retries=2)

# Get user input
user_text = input("Enter text to translate to French: ")

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", user_text),
]

ai_msg = llm.invoke(messages)
print(f"\nTranslation: {ai_msg.content}")



#llm = Anthropic(model="Claude 3.5 Haiku", anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
#model.invoke("what is the capital of france")
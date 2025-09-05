import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.sequential import SequentialChain

load_dotenv(find_dotenv())

model = ChatAnthropic(model="claude-sonnet-4-20250514",
                      temperature= 0.5,
                      max_tokens=2000,
                      max_retries=2)

template = """
As a comedian, please come up with a simple funny quote. Max words = 90
Please use this topic {topic} and this character {character} to generate the quote


Quote
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model | StrOutputParser()


template_for_transalation = """
Translate the {funny_quote} in {language}. Make sure the language is simple
"""

prompt_for_translation = ChatPromptTemplate.from_template(template_for_transalation)
chain_for_translation = prompt_for_translation | model | StrOutputParser()

# Get user input
topic = input("Enter a topic: ")
character = input("Enter a character: ")
language = input("Enter target language: ")

# Generate the funny quote
funny_quote = chain.invoke({"topic": topic, "character": character})

# Translate it
translated_quote = chain_for_translation.invoke({"funny_quote": funny_quote, "language": language})

# Beautiful output
print("\n" + "="*60)
print("🎭 FUNNY QUOTE GENERATOR 🎭")
print("="*60)
print(f"📝 Topic: {topic}")
print(f"🎬 Character: {character}")
print(f"🌍 Language: {language}")
print("\n" + "-"*60)
print("💬 ORIGINAL QUOTE:")
print(f'   "{funny_quote}"')
print("\n" + "-"*60)
print(f"🌐 TRANSLATED QUOTE ({language.upper()}):")
print(f'   "{translated_quote}"')
print("="*60)
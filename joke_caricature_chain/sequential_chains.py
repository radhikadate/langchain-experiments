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


template_for_caricature = """
Based on this funny quote: "{funny_quote}"

Create a detailed caricature description that visually represents this joke. Include:
1. The character's exaggerated features and expression
2. Visual elements that illustrate the humor
3. Props, setting, or background details
4. Specific artistic style suggestions

Make it vivid and detailed enough for an artist to draw.

Caricature Description:
"""

prompt_for_caricature = ChatPromptTemplate.from_template(template_for_caricature)
chain_for_caricature = prompt_for_caricature | model | StrOutputParser()

# Get user input
topic = input("Enter a topic: ")
character = input("Enter a character: ")

# Generate the funny quote
funny_quote = chain.invoke({"topic": topic, "character": character})

# Generate caricature description
caricature_description = chain_for_caricature.invoke({"funny_quote": funny_quote})

# Beautiful output
print("\n" + "="*70)
print("🎭 JOKE & CARICATURE GENERATOR 🎨")
print("="*70)
print(f"📝 Topic: {topic}")
print(f"🎬 Character: {character}")
print("\n" + "-"*70)
print("💬 FUNNY JOKE:")
print(f'   "{funny_quote}"')
print("\n" + "-"*70)
print("🎨 CARICATURE DESCRIPTION:")
print(f"   {caricature_description}")
print("="*70)
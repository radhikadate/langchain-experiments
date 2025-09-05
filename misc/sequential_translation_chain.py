import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_core import prompts
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
#from langchain.chains import LLMChain
load_dotenv(find_dotenv())


model = ChatAnthropic(model="claude-sonnet-4-20250514",
                      temperature= 0.5,
                      max_tokens=2000,
                      max_retries=2)

""" prompt = PromptTemplate(
    input_variables=["language"],
    template= "How do you say good morning in {language}?")
chain = LLMChain(llm=model, prompt=prompt) """

##print(chain.invoke(language ="Spanish"))


############  With Runnable #################################
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


first_prompt = ChatPromptTemplate.from_template("How can I say {text} in {language}")
first_chain = first_prompt | model | StrOutputParser()


second_prompt = ChatPromptTemplate.from_template("What are two other ways of saying {say} without changing the meaning?")
second_chain = second_prompt | model 

sequential_chain = (
    {"say": first_chain, "text": RunnablePassthrough(), "language": RunnablePassthrough()}
    | second_chain
)

# Get user input
english_phrase = input("Enter an English phrase: ")
language = input("Enter target language: ")

print(sequential_chain.invoke({"text": english_phrase, "language": language}).content)

#This worked when all the variables were removed from the first prompt
#sequential_chain = (
#                    {"say" : first_chain}
#                    |second_chain|model
#                   )
#print(first_chain.invoke(input = {}))
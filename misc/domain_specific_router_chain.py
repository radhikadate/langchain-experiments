import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.sequential import SequentialChain
from langchain.chains import LLMChain

load_dotenv(find_dotenv())

model = ChatAnthropic(model="claude-sonnet-4-20250514",
                      temperature= 0.5,
                      max_tokens=2000,
                      max_retries=2)

biology_template = """You are a very smart biology professor
You are great at answering the questions abut biology in a concise and easy to understand manner
When you dont know the answer, you admit that you dont know.

Here is a question:
{input}"""

math_template ="""You are a very good mathematician. 
You are good at answering the questions because you are able to break down the hard problems into components.
You first answer the component parts and then put them togather to answer the broader question

Here is the question:
{input}"""

astronomy_template = """You are a very good astronomer. You are great at answering the question.
You first answer the component parts and then put them togather to answer the broader question

Here is the question:
{input}
"""

travel_agent_template = """You are a very good travel agent with a large amount of knowledge 
when it comes to getting paople the best deals and recommendations for travel, vacations,
flights to the worlds's best destinations.
You are good a good tourist guide who plans itineries well:

Here is a question:
{input}
"""

prompt_infos = [
    {
        "name": "Biology",
        "description": "Good at answering biology questions",
        "prompt_template":biology_template
    },
    {
        "name": "Math",
        "description": "Good at answering math questions",
        "prompt_template":math_template   
    },
    {
        "name": "Astronomy",
        "description": "Good at answering astronomy questions",
        "prompt_template":astronomy_template
    },
    {
        "name": "Travel Agent",
        "description": "Good at answering travel questions",
        "prompt_template":travel_agent_template
    }
]
destination_chains = {}
for info in prompt_infos:
    name = info["name"]
    prompt_template = info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm= model, prompt=prompt)
    destination_chains[name] = chain
    
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chaining = LLMChain(llm= model, prompt=default_prompt)

from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router import MultiPromptChain

destinations = [f"{p['name']}:{p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
    )

router_chain = LLMRouterChain.from_llm(
    model, router_prompt
)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chaining,
    verbose=True
)

user_question = input("Enter your question: ")
response = chain.invoke(user_question)

print("\n" + "="*50)
print("EXPERT RESPONSE")
print("="*50)
print(response['text'])
print("\nRouted to:", response.get('destination', 'Default Chain'))
print("="*50)
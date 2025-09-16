from dotenv import load_dotenv , find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import Tool, create_tool_calling_agent, AgentExecutor, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_community.docstore import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

load_dotenv(find_dotenv())
model = ChatAnthropic(model="claude-sonnet-4-20250514",
                      temperature= 0.5,
                      max_tokens=2000,
                      max_retries=2)

docstore = DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to answer questions about people, places or facts. Be specific"
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask about a specific person, place or fact. Ask targetted questions"
    )       
]

# Initialize the agent
docstore_agent = initialize_agent(
    tools,
    model,
    agent="react-docstore",
    verbose=True,
    max_iterations=3
)

user_query = input("Enter your question: ")
result = docstore_agent.run(user_query)

# Beautiful output formatting
print("\n" + "="*80)
print("📚 WIKIPEDIA DOCSTORE AGENT RESPONSE")
print("="*80)
print(f"❓ Question: {user_query}")
print("\n" + "-"*80)
print("📖 ANSWER:")
print(result)
print("="*80)
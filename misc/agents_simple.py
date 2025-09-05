from dotenv import load_dotenv , find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import tool, create_tool_calling_agent, AgentExecutor, load_tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
import requests
from datetime import datetime


@tool
def add(x:float, y:float) -> float:
    """Add 'x' and 'y'."""
    return x+y

@tool
def subtract(x:float, y:float) -> float:
    """Add 'x' and 'y'."""
    return x-y

@tool
def exponentiate(x:float, y:float) -> float:
    """Raise 'x' to the power 'y'"""
    return x**y

@tool 
def multiply(x:float, y:float) -> float:
    """Multiply 'x' by 'y'"""
    return x*y

@tool
def get_location_from_ip():
    """Get geographical location from ip address"""
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        if 'loc' in data:
            latitude, longitude = data['loc'].split(',')
            data = (
                f"Latitude: {latitude},\n"
                f"Longitude: {longitude},\n"
                f"City: {data.get('city','N/A')},\n"
                f"Country: {data.get('country', 'N/A')}"
                )
            return data
        else:
            return "Location data not available"
    except Exception as e:
            return f"An error occurred: {e}"

@tool
def get_current_date_time() -> str:
    """Get current date and time"""
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"Current date and time: {current_datetime}"




        
load_dotenv(find_dotenv())

model = ChatAnthropic(model="claude-sonnet-4-20250514",
                      temperature= 0.5,
                      max_tokens=2000,
                      max_retries=2)

prompt = ChatPromptTemplate.from_messages([
("system", "You are a helpful AI bot."),
MessagesPlaceholder(variable_name="chat_history"),
("human", "{input}"),
("placeholder", "{agent_scratchpad}")
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

tools =[add, subtract, exponentiate, multiply]
toolbox = load_tools(tool_names=['serpapi'], llm=model)

agent = create_tool_calling_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
agent_executor.invoke({"input": "what is my name",
              "chat_history": memory.chat_memory.messages})


from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from typing import List, Dict, Any
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool


# Load environment variables from .env file
load_dotenv()


# Define pydantic model for the configuration
class ResearchResponse(BaseModel):
    topic: str = Field(..., description="The topic of the research")
    summary: str = Field(..., description="A summary of the research findings")
    source: list[str] = Field(..., description="Sources of the research findings")  
    tools_used: list[str] = Field(..., description="Tools used in the research process")
    
    
# define pydantic parser for the response
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Initialize the chat model with Cohere
chat = ChatCohere()

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a highly skilled research assistant. Your task is to provide detailed and well-structured research output.
            Use the necessary tools to gather information and ensure the response is comprehensive.
            Wrap the output in this format and provide no other text:\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query})


try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
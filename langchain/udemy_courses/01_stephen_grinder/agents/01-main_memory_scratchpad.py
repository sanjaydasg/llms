from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
	HumanMessagePromptTemplate, 
	ChatPromptTemplate, 
	MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from tools.sql import run_query_tool, list_tables, desctibe_tables_tool
from tools.report import write_report_tool

from dotenv import load_dotenv
load_dotenv()


chat = ChatOpenAI()

tables = list_tables()

prompt = ChatPromptTemplate(
	messages=[	
		SystemMessage(content="You are an AI that has access to a SQLite database.\n"
							  f"The databse has tables of: {tables} \n"
							  "Do not make assumptions about what tables exist or what columns exist.\n"
							  "Instead use the 'describe_tables' function"),
		HumanMessagePromptTemplate.from_template("{input}"),
		# agent_scratchpad below is simalr to memory
		# the history of the conversation is captured on the scratchpad
		# the difference is that the scratch_pad saves intermediate function calls
		# and when an AI message is sent back, these intermediate messages get deleted
		# and the AI message is returned to the user. Succesive calls to the agent_executor 
		# do not have memory of previous calls 
		MessagesPlaceholder(variable_name="agent_scratchpad")
	]
)

tools = [
		run_query_tool, 
		desctibe_tables_tool, 
		write_report_tool
]

# agent is similar to a chain with the exception that it
# has additional functionality specified in tools
agent = OpenAIFunctionsAgent(
	llm=chat,
	prompt=prompt,
	tools=tools
)

agent_executor = AgentExecutor(
	agent=agent,
	verbose=True,
	tools=tools
)

#query = "How many users are there in the database?"
#query = "How many users have shipping addresses?"
query = "Summarize the top 5 most popular products. Write the results to a report."

agent_executor(query)

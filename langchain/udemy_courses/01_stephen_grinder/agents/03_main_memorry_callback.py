from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
	HumanMessagePromptTemplate, 
	ChatPromptTemplate, 
	MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory


from tools.sql import run_query_tool, list_tables, desctibe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler


from dotenv import load_dotenv
load_dotenv()

handler = ChatModelStartHandler()
chat = ChatOpenAI(
	callbacks=[handler]
)

tables = list_tables()

prompt = ChatPromptTemplate(
	messages=[	
		SystemMessage(content="You are an AI that has access to a SQLite database.\n"
							  f"The databse has tables of: {tables} \n"
							  "Do not make assumptions about what tables exist or what columns exist.\n"
							  "Instead use the 'describe_tables' function"),
		# we place this placeholder for memory before any human messages are passed
		MessagesPlaceholder(variable_name="chat_history"),
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

memory = ConversationBufferMemory(
				memory_key="chat_history", 
				return_messages=True
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
	# verbose=True, 	# commenting this out with use of handler
	tools=tools,
	memory=memory
)


query = "How many users are there in the database?"
agent_executor(query)

query = "How many users have shipping addresses?"
agent_executor(query)


query = "Summarize the top 5 most popular products. Write the results to a report."
agent_executor(query)

query = "Repeat the exact process for users"
agent_executor(query)



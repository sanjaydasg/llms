from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, FileChatMessageHistory

import argparse

from dotenv import load_dotenv
load_dotenv()


parser = argparse.ArgumentParser()
parser.add_argument("--memory", default="summary")
parser.add_argument("--verbose", default=True)

args = parser.parse_args()

bln_verbose = args.verbose
chat = ChatOpenAI(verbose=bln_verbose)

if args.memory!='summary':
	memory = ConversationBufferMemory(
		chat_memory=FileChatMessageHistory("messages.json"),
		memory_key="messages",
		return_messages=True
	)
else:
	memory = ConversationSummaryMemory(
		#chat_memory=FileChatMessageHistory("messages.json"),
		memory_key="messages",
		return_messages=True,
		llm=chat
	)


prompt = ChatPromptTemplate(
	input_variables = ["content", "messages"],
	messages = [
	MessagesPlaceholder(variable_name="messages"),
	HumanMessagePromptTemplate.from_template("{content}")
	]
)

chain = LLMChain(
	llm=chat,
	prompt=prompt,
	memory=memory,
	verbose=bln_verbose
	)

while True:

	content = input(">> ")
	result = chain({"content": content})
	print(result["text"])


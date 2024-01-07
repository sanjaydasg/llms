from langchain.callbacks.base import BaseCallbackHandler
from pyboxen import boxen


def boxen_print(*args, **kwargs):
	print(boxen(*args, **kwargs))

# test
#boxen_print("Hello", title="test", color="yellow")

class ChatModelStartHandler(BaseCallbackHandler):
	def on_chat_model_start(self, serialized, messages, **kwargs):

		print("\n============== Sending Messages ==============")
		for message in messages[0]:
			if message.type=="system":
				print(message)
				#boxen_print(message.content, title=message.type, color="yellow")

			elif message.type=="human":
				boxen_print(message.content, title=message.type, color="green")

			# AI message that is trying to run a function
			elif message.type=="ai" and "function_call" in message.additional_kwargs:
				call = message.additional_kwargs["function_call"]
				boxen_print(f"Running tool {call['name']} with args {call['arguments']}", 
							title=message.type, 
							color="cyan"
				)

			# final AI message from llm
			elif message.type=="ai":
				boxen_print(message.content, title=message.type, color="blue")

			elif message.type=="function":
				boxen_print(message.content, title=message.type, color="purple")

			else: # in case there are any other message types in the future
				boxen_print(message.content, title=message.type, color="white")
		
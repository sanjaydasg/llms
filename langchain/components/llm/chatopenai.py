from langchain.chat_models import ChatOpenAI

def build_llm(llm_args, llm_model):
	return ChatOpenAI(
		streaming=llm_args["streaming"],
		model_name=llm_model,
		temperature=llm_args["temperature"]
	)

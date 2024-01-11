from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader

from string import Template

from components.llm import llm_map

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

chat_args = {"streaming": False,
			 "temperature": 0
}

chat_model = "gpt-3.5"
llm = llm_map[chat_model](chat_args)


text= """
Mojo combines the usability of Python with the performance of C, unlocking unparalleled programmability \
of AI hardware and extensibility of AI models.
Mojo is a new programming language that bridges the gap between research and production \ 
by combining the best of Python syntax with systems programming and metaprogramming.
With Mojo, you can write portable code that’s faster than C and seamlessly inter-op with the Python ecosystem.
When we started Modular, we had no intention of building a new programming language. \
But as we were building our platform with the intent to unify the world’s ML/AI infrastructure, \
we realized that programming across the entire stack was too complicated. Plus, we were writing a \
lot of MLIR by hand and not having a good time.
And although accelerators are important, one of the most prevalent and sometimes overlooked "accelerators" \
is the host CPU. Nowadays, CPUs have lots of tensor-core-like accelerator blocks and other AI acceleration \
units, but they also serve as the “fallback” for operations that specialized accelerators don’t handle, \
such as data loading, pre- and post-processing, and integrations with foreign systems. \
"""


# A. Basic Prompt

system_message = 'You are an expert copywriter with expertize in summarizing documents'
human_message_template = Template('Please provide a short and concise summary of the following text:\n TEXT: $text')


def basic_prompt(llm, 
				 system_message, 
				 human_message_template, 
				 text
	):
	
	human_message = human_message_template.substitute({'text': text})
	messages = [
    	SystemMessage(content=system_message),
    	HumanMessage(content=human_message)
	]

	summary_output = llm(messages)
	return summary_output.content

#summarized_text = basic_prompt(llm, system_message, human_message_template, text)
#print(summarized_text)


# B. Prompt Template


message_template = '''Write a concise and short summary of the following text: 
					  TEXT: {text} Translate the summary to {language}.'''

input_variable_dict = {'text': text,
					   'language': 'hindi'
}

def basic_prompt_template(llm, message_template, input_variable_dict):

	prompt = PromptTemplate(
	    input_variables=list(input_variable_dict.keys()),
	    template=message_template
	)

	chain = LLMChain(llm=llm, 
					 prompt=prompt
	)

	summarized_text = chain.run(input_variable_dict)
	return summarized_text

#summarized_text = basic_prompt_template(llm, message_template, input_variable_dict)
#print(summarized_text)


# More advanced summarizations:

def summarize(chain_inputs, text_inputs, prompt_inputs={}):
	"""_summary_

	Args:
		chain_inputs (dictionary): llm, verbose, chain_type
		text_inputs (dictionary): chunks, text, or doc
		prompt_inputs (dict, optional): message and refine prompt Defaults to {}.

	Returns:
		text: summary of provided text_input
	"""	
	chain_type = chain_inputs['chain_type']
	if prompt_inputs == {}:
		chain = load_summarize_chain(**chain_inputs)

	elif chain_type == 'stuff':
		prompt = PromptTemplate(**prompt_inputs)
		chain_inputs = dict(**{**chain_inputs, **{'prompt': prompt}})

	elif chain_type == 'map_reduce':
		map_prompt = PromptTemplate(**prompt_inputs['map'])
		combine_prompt = PromptTemplate(**prompt_inputs['combine'])
		chain_inputs = dict(**{**chain_inputs, 
						       **{'map_prompt': map_prompt},
						       **{'combine_prompt': combine_prompt}
		})

	elif chain_type == 'refine':
		question_prompt = PromptTemplate(**prompt_inputs['question'])
		refine_prompt = PromptTemplate(**prompt_inputs['refine'])
		chain_inputs = dict(**{**chain_inputs, 
						       **{'question_prompt': question_prompt},
						       **{'refine_prompt': refine_prompt}
		})
	else:
		Print("Chain type not supported. Please use stuff, map_reduce or refine")
		return

	chain = load_summarize_chain(**chain_inputs)
	summarized_text = chain.run(text_inputs)
	return summarized_text

# C. Stuff Document Chain

file_name = './files/sj.txt'
with open(file_name, encoding='utf-8') as f:
    text = f.read()

docs = [Document(page_content=text)]

message_template = '''Write a concise and short summary of the following text: 
					  TEXT: {text} 
				   '''

input_dict = {
		'llm': llm,
		'verbose': False,
		'chain_type': 'stuff'
}

prompt_dict = {
		'input_variables': ['text'],
		'template': message_template
}

'''
summarized_text = summarize(
					prompt_inputs=prompt_dict,
					chain_inputs=input_dict,
					text_inputs=docs
)
print(summarized_text)
'''
# D.a map_reduce

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
chunks = text_splitter.create_documents([text])

input_dict = {
		'llm': llm,
		'verbose': False,
		'chain_type': 'map_reduce'
}

'''
summarized_text = summarize(
					chain_inputs=input_dict,
					text_inputs=chunks
)
print(summarized_text)
'''

# D.b map_reduce with custom prompt



map_template = """Write a short and concise summary of the following: 
				  Text: '{text}' CONCISE SUMMARY."""


combine_template = """Write a concise summary of the following text that covers the key points.
					  Add a title to the summary.
					  Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
					  by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.
					  Text: '{text}'"""

input_dict = {
		'llm': llm,
		'verbose': False,
		'chain_type': 'map_reduce'
}

prompt_dict = {'map':
					{'input_variables': ['text'],
					 'template': map_template},
				'combine':
					{'input_variables': ['text'],
					 'template': combine_template}
}

'''
summarized_text = summarize(
					prompt_inputs=prompt_dict,
					chain_inputs=input_dict,
					text_inputs=chunks
)
print(summarized_text)
'''

# E.a Refine Chain
loader = UnstructuredPDFLoader('./files/attention-is-all-you-need.pdf')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

input_dict = {
		'llm': llm,
		'chain_type': 'refine',
		'verbose': False
}

'''
summarized_text = summarize(
					chain_inputs=input_dict,
					text_inputs=chunks
)

print(summarized_text)
'''

# E.b Refine Chain with custom prompt



prompt_template = """Write a short and concise summary of the following: 
				  Text: '{text}' CONCISE SUMMARY."""


refine_template = """
    Your job is to produce a final summary.
    I have provided an existing summary up to a certain point: {existing_answer}.
    Please refine the existing summary with some more context below.
    ------------
    {text}
    ------------
    Start the final summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
    by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.
    
"""

input_dict = {
		'llm': llm,
		'return_intermediate_steps': False,
		'chain_type': 'refine'
}

prompt_dict = {'question':
					{'input_variables': ['text'],
					 'template': prompt_template},
				'refine':
					{'input_variables': ['existing_answer', 'text'],
					 'template': refine_template}
}


summarized_text = summarize(
					prompt_inputs=prompt_dict,
					chain_inputs=input_dict,
					text_inputs=chunks
)
print(summarized_text)



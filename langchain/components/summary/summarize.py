from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain


def summarize_text(chain_inputs, text_inputs, prompt_inputs={}):
	
	"""
	This function takes in inputs for creating a chain and prompt and then summarizes the input_text
	using one of the three methodologies: stuff-document, map-reduce, or refine

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



def build_summary_input_map(llm, chain_type, verbose):

	"""
	This function prepares the inputs that are required for the three types of summarizations

	Args:
		llm (chat model): llm that will be used for summarization
		chain_type (string): stuff, map_reduce or refine
		verbose (boolean): True for describing intermediate steps

	Returns:
		dict: dictionary containing input_dict for chain and prompt dict for prompts
	"""
	if chain_type == 'stuff':
		message_template = '''Write a concise and short summary of the following text: 
							  TEXT: {text} 
						   '''
		input_map = {
					'input_dict' : {
							'llm': llm,
							'verbose': verbose,
							'chain_type': 'stuff'
					},
					'prompt_dict' : {
							'input_variables': ['text'],
							'template': message_template
					}
			}

	elif chain_type == 'map_reduce':

		map_template = """Write a short and concise summary of the following: 
						  Text: '{text}' CONCISE SUMMARY.
		"""


		combine_template = """Write a concise summary of the following text that covers the key points.
							  Add a title to the summary.
							  Start your summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
							  by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.
							  Text: '{text}'
		"""

		input_map = {
					'input_dict': {
							'llm': llm,
							'verbose': verbose,
							'chain_type': 'map_reduce'
					},
					'prompt_dict': {'map':
										{'input_variables': ['text'],
										 'template': map_template},
									'combine':
										{'input_variables': ['text'],
										 'template': combine_template}
					}
			}

	elif chain_type == 'refine':

		prompt_template = """Write a short and concise summary of the following: 
						     Text: '{text}' CONCISE SUMMARY.
		"""


		refine_template = """Your job is to produce a final summary.
		    				  I have provided an existing summary up to a certain point: {existing_answer}.
						      Please refine the existing summary with some more context below.
						      ------------
						      {text}
						      ------------
						      Start the final summary with an INTRODUCTION PARAGRAPH that gives an overview of the topic FOLLOWED
						      by BULLET POINTS if possible AND end the summary with a CONCLUSION PHRASE.
		    
		"""
		input_map = {
					'input_dict': {
							'llm': llm,
							'return_intermediate_steps': verbose,
							'chain_type': 'refine'
					},
					'prompt_dict': {'question':
										{'input_variables': ['text'],
										 'template': prompt_template},
									'refine':
										{'input_variables': ['existing_answer', 'text'],
										 'template': refine_template}
					}
			}

	return input_map
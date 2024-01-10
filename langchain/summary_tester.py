
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from components.llm import llm_map
from components.summary.summarize import build_summary_input_map, summarize_text
from components.reader.document import chunk_document
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


llm_args = {"streaming": False,
			 "temperature": 0
}
llm_model = "gpt-3.5"
chain_type='refine'
file_name = './files/sj.txt'
pfd_file_name = './files/attention-is-all-you-need.pdf'

llm = llm_map[llm_model](llm_args)
input_map  = build_summary_input_map(llm, chain_type, False)

chunks = chunk_document('pdf', pfd_file_name)
summarized_text = summarize_text(
				chain_inputs=input_map['input_dict'],
                text_inputs=chunks,
                prompt_inputs=input_map['prompt_dict']
)

print(summarized_text)
from .chatopenai import build_llm
from functools import partial

llm_map = {
	"gpt-4": partial(build_llm, llm_model="gpt-4"),
	"gpt-3.5": partial(build_llm, llm_model="gpt-3.5-turbo")
}
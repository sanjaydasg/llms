
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()

# find relevant facts
# context and embeddings

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
	separator="\n",
	chunk_size=200,
	chunk_overlap=0)

loader = TextLoader("facts.txt")
# docs = loader.load() # to only load text
docs = loader.load_and_split(
		text_splitter=text_splitter)

db = Chroma.from_documents(
	docs,
	embedding=embeddings,
	persist_directory="emb"
)

query = "What is an interesting fact about the English Language?"
results = db.similarity_search_with_score(query)

for result in results:
	print("\n")
	print(result[1])
	print(result[0].page_content)
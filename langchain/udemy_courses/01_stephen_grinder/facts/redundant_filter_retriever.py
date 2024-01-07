from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever


class RedundantFilterRetriever(BaseRetriever):

	embeddings: Embeddings
	chroma: Chroma


	def get_relevant_documents(self, query):
		# given a query return a list of documents
		# this is a synchronous call

		#1. calculate embeddings for the query
		emb = self.embeddings.embed_query(query)	

		#2. take embeddings and feed them into
		# max_marginal_relevance_search_by_vector function

		return self.chroma.max_marginal_relevance_search_by_vector(
			embedding=emb,
			lambda_mult=0.6)

	async def aget_relevant_documents(slef):
		# this is an asynchronous call, that is functionally required
		# we will define it but keep it empty

		return []

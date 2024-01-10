from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader



def chunk_document(doc_type: str, 
                   file_path: str,
                   chunk_size=10000,
                   chunk_overlap=50
):
    """
    Create chunks from a given document

    Args:
        doc_type (str): type of document: 'text', or 'pdf'
        file_path (str): path for file
        chunk_size (int, optional):  Defaults to 10000.
        chunk_overlap (int, optional):  Defaults to 50.

    Returns:
        chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap
    )

    if doc_type=='text':

        with open(file_path, encoding='utf-8') as f:
            data = f.read()
        
        chunks = text_splitter.create_documents([data])

    elif doc_type=='pdf':

        loader = UnstructuredPDFLoader(file_path)
        data = loader.load()
        chunks = text_splitter.split_documents(data)

    return chunks
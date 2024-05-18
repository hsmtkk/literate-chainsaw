import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings


CHUNK_SIZE = 1024
CHUNK_OVERLAP = CHUNK_SIZE / 2

embeddings = OpenAIEmbeddings()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

faiss = None

for name in os.listdir("pdf"):
    path = os.path.join("pdf", name)
    loader = PyPDFLoader(path)
    docs = loader.load_and_split(splitter)
    if faiss is None:
        faiss = FAISS.from_documents(docs, embeddings)
    else:
        faiss.add_documents(docs)

faiss.save_local("faiss")

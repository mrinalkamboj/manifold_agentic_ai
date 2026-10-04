"""
pip install -qU langchain-postgres
docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
"""
from langchain_postgres.vectorstores import PGVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.indexing.base import InMemoryRecordManager
from langchain_core.indexing import index


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
record_manager = InMemoryRecordManager(namespace="langchain_demo")

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain" 

collection_name = "my_docs"

record_manager.create_schema()

cats_loader = TextLoader("cats.txt")
dogs_loader = TextLoader("dogs.txt")
cats_new_loader = TextLoader("cats_new.txt")

cats_docs = cats_loader.load()
dogs_docs = dogs_loader.load()
cats_new_docs = cats_new_loader.load()

print(cats_docs)
print(dogs_docs)
print(cats_new_docs)

vector_store = PGVector( embeddings=embeddings, 
collection_name=collection_name, 
connection=connection, use_jsonb=True, )

print("Loading the documents into the vector store")
print(cats_docs)

index_1 = index(cats_docs,record_manager, vector_store, source_id_key = "source")
print(f"Indexing the cats documents: {index_1}")

index_2 = index(dogs_docs,record_manager, vector_store, source_id_key = "source")
print(f"Indexing the dogs documents: {index_2}")

index_3 = index(cats_new_docs,record_manager, vector_store, source_id_key = "source")
print(f"Indexing the cats new documents: {index_3}")

# Repeat Index the documents of cats_docs
index_4 = index(cats_docs,record_manager, vector_store, source_id_key = "source")
print(f"Indexing the cats documents again: {index_4}")


# modify the document
with open("cats.txt", "w") as f:
    f.write("I love dogs not cats")

cats_docs = cats_loader.load()
print(cats_docs)

index_5 = index(cats_docs,record_manager, vector_store, source_id_key = "source")
print(f"Indexing the cats documents again: {index_5}")



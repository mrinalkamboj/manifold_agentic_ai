import pdfplumber
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json
pdf_path = Path(__file__).parent/"test_documents"/ "1810.04805v2.pdf"
import hashlib
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from dotenv import load_dotenv
load_dotenv()
import os
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")


def extract_text_from_pdf(pdf_path: Path) -> str:
    
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text(use_text_flow=True)
        return text

text = extract_text_from_pdf(pdf_path)
print(text)

# # Chunking the text

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10, separators=["\n\n", "\n", ". ", " ", ""])
# chunks = text_splitter.split_text(text)
# print("--------------------------------")
# print(len(chunks))
# print("--------------------------------")
# # Print the first 10 chunks
# for chunk in chunks:
#     print(chunk)
#     print("--------------------------------")
# Create Document Object
# documents = [Document(page_content=chunk, metadata={"source": pdf_path.name}) for chunk in chunks]
# print(documents)
# print("--------------------------------")
# print(len(documents))
# print("--------------------------------")

# # Save the documents to a JSON file
# with open("documents.json", "w") as f:
#     json.dump(documents, f)

chunk_example = Path(__file__).parent/"test_documents"/ "02_legal_contract.txt"

text = chunk_example.read_text(encoding="utf-8")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", ". ", " ", ""])

chunks = text_splitter.split_text(text)
print(len(chunks))
print("--------------------------------")
# Print the first 10 chunks
for chunk in chunks:
    print(f"Chunk: {chunk} \n chunk_length: {len(chunk)}")
    print("--------------------------------")

# Create Document Object
documents = [Document(page_content=chunk, metadata={"source": chunk_example.name, "chunk_id": hashlib.sha256(chunk.encode()).hexdigest()}) for chunk in chunks]
print(documents)
print("--------------------------------")
print(len(documents))
print("--------------------------------")

# # Save the documents to a JSON file
# with open("documents.json", "w") as f:
#     json.dump(documents, f)

# example hashlib.sha256(

t1 = 'Hello, world!'
t2 = 'Hello, world!'
t3 = 'Hello, world! Hello, world!'
t4 = 'Hello, world! '

h1 = hashlib.sha256(t1.encode()).hexdigest()
h2 = hashlib.sha256(t2.encode()).hexdigest()
h3 = hashlib.sha256(t3.encode()).hexdigest()
h4 = hashlib.sha256(t4.encode()).hexdigest()
print(h1)
print("--------------------------------")
print(h2)
print("--------------------------------")
print(h3)
print("--------------------------------")
print(h1 == h2)
print(h1 == h3)
print(h2 == h3)
print(h1 == h4)
print("--------------------------------")
# Embedding the chunks
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = embedding_model.embed_documents(chunks)
for embedding in embeddings[:3]:
    print(embedding)
    print("--------------------------------")
    print(f"len(embeddings): {len(embedding)}")
    print("--------------------------------")

# # Save the embeddings to a JSON file
with open("embeddings.json", "w") as f:
    json.dump(embeddings, f)

# Load the embeddings to VectorStore

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain" 

collection_name = "my_docs"

vector_store = PGVector( embeddings=OpenAIEmbeddings(model="text-embedding-3-small"), 
collection_name=collection_name, 
connection=connection, use_jsonb=True, )

vector_store.add_documents(documents=documents, ids=[doc.metadata["chunk_id"] for doc in documents])

# Query the vector store
query = "what is the Liability and Damages from the contract?"

results = vector_store.similarity_search(query,k=5)

for result in results:
    print(result.page_content)
    print(result.metadata)
    # print(f"similarity: {result.similarity_score}")
    print("--------------------------------")


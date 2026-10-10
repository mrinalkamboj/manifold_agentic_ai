from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
load_dotenv()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)


documents = SimpleDirectoryReader(input_dir="./data",required_exts=[ ".txt"]).load_data()
print(documents)

index = VectorStoreIndex.from_documents(documents)
print("Vector Index created")
#1.  process the documents into nodes
#2. Generate embeddings for the nodes
#3. builds a vector store index

question = "can an AI Agent Approve a refund for INR 18,000 ?"

retriever = index.as_retriever( similarity_top_k=2)

results = retriever.retrieve(question)
print("*******************Results*******************")
for i, result in enumerate(results, 1):
    print(f"\nTop {i} result:")
    print(result.text)
    print(f"Score: {result.score}")
    print(f"filename: {result.metadata['file_name']}")
    print("-" * 50)

# llama query engine

query_engine = index.as_query_engine(
    similarity_top_k=2,
)

response = query_engine.query(question)
print("*******************Response*******************")
print(response)

print("Retrieved Sources:")
for source in response.source_nodes:
    print(f"Filename: {source.metadata['file_name']}")
    print(f"Similarity Score: {source.score}")
    print("-" * 50)
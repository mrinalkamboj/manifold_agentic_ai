from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")

chain = prompt | llm

response = chain.invoke({"country": "France"})
print(response)

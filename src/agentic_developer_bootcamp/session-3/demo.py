# pip install -r requirements.txt

from fastapi import FastAPI # import the FastAPI class from the fastapi module

app = FastAPI() # create an instance of the FastAPI class

@app.get("/")
def greeting():
    return "Mrinal is serving the Rest API with Fast API"

@app.post("/add")
def add(a: float, b: float):
    return a + b

@app.post("/subtract")
def subtract(a: float, b: float):
    return a - b

@app.post("/multiply")
def multiply(a: float, b: float):
    return a * b

# To Run the application, use the following command:
# uvicorn demo:app --reload
from langchain_core.runnables import RunnableLambda

def add_ten(number):
    return number + 10

def multiply_by_two(number):
    return number * 2


step_1 = RunnableLambda(add_ten) # converts the python function into a Langchain RunnableLambda object
step_2 = RunnableLambda(multiply_by_two)

chain = step_1 | step_2

print(chain.invoke(5))
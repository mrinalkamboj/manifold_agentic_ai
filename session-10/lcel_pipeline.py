from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

from common import insufficient_result, pack_answer
from rag_core.prompts import build_messages
from rag_core.providers import make_answer_model, offline_answer
from rag_core.rag import retrieve
from rag_core.schemas import Question


def build_chain(settings, offline: bool = False):
    validate_question = RunnableLambda(lambda question: Question(text=question).text)
    collect_evidence = RunnableLambda(lambda question: {
        "question": question, "hits": retrieve(question, settings, offline)})
    if offline:
        generate = RunnableLambda(lambda data: offline_answer(data["hits"]))
    else:
        generate = RunnableLambda(lambda data: build_messages(data["question"], data["hits"])) | make_answer_model(settings)
    answer_and_check = (
        RunnablePassthrough.assign(answer=generate)
        | RunnableLambda(lambda data: pack_answer(data["answer"], data["hits"], offline))
    )
    choose_answer = RunnableBranch(
        (lambda data: not data["hits"], RunnableLambda(lambda data: insufficient_result(offline))),
        answer_and_check,
    )
    return validate_question | collect_evidence | choose_answer


def run_chain(question, settings, offline: bool = False) -> dict:
    return {"pipeline": "lcel", **build_chain(settings, offline).invoke(question)}

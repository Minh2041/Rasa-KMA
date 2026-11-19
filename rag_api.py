
import logging

from fastapi import FastAPI
from pydantic import BaseModel
from rag_kma import generate_answer

app = FastAPI()
logger = logging.getLogger(__name__)


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_rag(data: Question):
    question = data.question.strip()

    if not question:
        return {
            "answer": "",
            "answered": False,
            "sources": []
        }

    logger.warning("RAG question: %s", question)

    # Sử dụng logic phân loại folder đã viết
    result = generate_answer(question)

    sources = []

    for source in result.sources:
        metadata = source.metadata or {}

        sources.append({
            "category": metadata.get("category"),
            "source": metadata.get("source"),
            "content": source.page_content[:200]
        })

    logger.warning("RAG sources: %s", sources)

    return {
        "answer": result.answer,
        "answered": bool(sources),
        "sources": sources
    }


@app.get("/health")
def health():
    return {"status": "ok"}

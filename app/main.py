from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from pydantic import BaseModel

from .graph import build_graph


class QuestionRequest(BaseModel):
    question: str

app = FastAPI(title="LangGraph Wikipedia Agent")

# CORS, чтобы index.html мог ходить к API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Компилируем граф один раз при старте
compiled_graph = build_graph()

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    """
    Обрабатывает запрос к агенту

    Параметры:
        req: Объект с текстом вопроса

    Возвращает:
        JSON с итоговым ответом и полным состоянием графа
    """
    result_state = await compiled_graph.ainvoke({"question": req.question})
    
    return {
        "answer" : result_state.get("final_answer", "Ответ не получен."),
        "state" : result_state
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

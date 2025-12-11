from typing import Dict, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import re
import math

from .model import cross_encoder
from .llm import run_llm_system_prompt
from .wiki_client import search_wiki, get_page_extract


class SearchQueryObj(TypedDict):
    query: str
    info_to_find: str


class AgentState(TypedDict, total=False):
    question: str
    search_queries: List[SearchQueryObj]

    wiki_pages: Dict[str, List[str]]
    wiki_best_chunks: Dict[str, str]
    final_answer: str

    answer_ok: bool
    answer_feedback: str
    original_final_answer: str


def node_extract_search_queries(state: AgentState) -> AgentState:
    """Извлекает из вопроса до двух запросов к Википедии и ключевые слова для поиска по тексту"""
    
    question = state["question"]

    system_prompt = """
Ты помощник, который разбивает пользовательский вопрос на ОТ 0 ДО 2 поисковых запросов в Википедию, которые помогут ответить на вопрос.
Сколько запросов генерировать:
- 0 - если вопрос не требует обращения к Википедии
- 1 - если для ответа нужна информация только об одном объекте
- 2 - если для ответа требуются два независимых запроса

Для каждого запроса выдели название страницы Википедии и информацию, которую необходимо найти на странице.

Например, вопрос: Сколько яблок помещается в ведре?
Нужно две страницы: "ведро" и "яблоко".
При поиске на странице про ведро нам нужно найти "объем ведра".
При поиске на странице про яблоко - "размер яблока".

Ответ выводи ТОЛЬКО список объектов в формате JSON: 
[
  {"query": "ведро", "info_to_find": "объем ведра"},
  {"query": "яблоко", "info_to_find": "размер яблока"}
]
"""

    raw = run_llm_system_prompt(system_prompt, question)

    try:
        import json
        query_objs: List[SearchQueryObj] = json.loads(raw)
    except Exception:
        # если LLM выдала невалидный JSON, fallback: каждая строка как запрос и ключевое слово целиком
        query_objs = [
            {"query": line.strip(), "info_to_find": line.strip()}
            for line in raw.splitlines()
            if line.strip()
        ]

    return {**state, "search_queries": query_objs}


async def node_fetch_wiki_pages(state: AgentState) -> AgentState:
    """Запрашивает страницы Википедии по каждому поисковому запросу"""
    
    search_queries: List[SearchQueryObj] = state["search_queries"]
    wiki_pages: Dict[str, List[str]] = {}

    for q_obj in search_queries:
        query_text = q_obj["query"]
        pages_texts: List[str] = []

        # используем сам текст запроса как строку поиска
        search_results = await search_wiki(query_text, limit=1)  # берем не более limit страниц - можно менять но будет использоваться больше токенов

        for item in search_results:
            pageid = item.get("pageid")
            if pageid is None:
                continue
            text = await get_page_extract(pageid)
            pages_texts.append(text)

        wiki_pages[query_text] = pages_texts

    return {**state, "wiki_pages": wiki_pages}



MAX_CHUNKS = 80
MIN_CHARS = 200
MAX_SEGMENT = 512

def _strip_title(page_text: str) -> str:
    """Срезает короткий заголовок страницы (первая строка без точки)"""
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    if not lines:
        return ""
    first = lines[0]
    if len(first) < 40 and "." not in first:
        lines = lines[1:]
    return "\n".join(lines)


def _split_paragraphs(text: str) -> List[str]:
    """Делит текст на параграфы по двойным переводам строки"""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _split_long_paragraph(paragraph: str) -> List[str]:
    """
    Если абзац длинный — режем его на сегменты по предложениям,
    чтобы каждый был около MAX_SEGMENT символов
    """
    if len(paragraph) <= MAX_SEGMENT:
        return [paragraph]

    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= MAX_SEGMENT:
            current = sent if not current else current + " " + sent
        else:
            if len(current.strip()) >= MIN_CHARS:
                chunks.append(current.strip())
            current = sent

    if len(current.strip()) >= MIN_CHARS:
        chunks.append(current.strip())

    return chunks


def _build_chunks_from_paragraphs(paras: List[str]) -> List[str]:
    """Генерирует текстовые чанки из списка параграфов"""
    chunks: List[str] = []
    for p in paras:
        chunks.extend(_split_long_paragraph(p))
    return [c for c in chunks if len(c) >= MIN_CHARS]


def node_select_best_chunk(state: AgentState) -> AgentState:
    """
    Для каждого search_query:
      1) собирает чанки из страниц Википедии
      2) оценивает релевантность чанков к (question + info_to_find) через кросс-энкодер
      3) берёт top-3 чанка по скору кросс-энкодера
      4) отдаёт эти 3 чанка в LLM, чтобы он выбрал лучший один
      5) сохраняет финальный топ-1 чанк в wiki_best_chunks[query]
    """

    search_query_objs = state["search_queries"]
    wiki_pages = state["wiki_pages"]

    wiki_best_chunks: Dict[str, str] = {}

    for q_obj in search_query_objs:
        query = q_obj["query"]
        info_to_find = q_obj["info_to_find"]

        pages = wiki_pages.get(query, [])

        all_chunks: List[str] = []
        for page_text in pages:
            text = _strip_title(page_text)
            paras = _split_paragraphs(text)
            chunks = _build_chunks_from_paragraphs(paras)
            all_chunks.extend(chunks)

        if not all_chunks:
            wiki_best_chunks[query] = ""
            continue

        all_chunks = all_chunks[:MAX_CHUNKS]

        ce_query = f"Что нужно найти: {info_to_find}"
        pairs = [(ce_query, chunk_text) for chunk_text in all_chunks]

        scores = cross_encoder.predict(pairs)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        TOP_K = 5
        top_indices = [idx for idx, _ in indexed_scores[:TOP_K]]
        top_chunks = [all_chunks[i] for i in top_indices]

        if len(top_chunks) == 1:
            wiki_best_chunks[query] = top_chunks[0]
            continue

        system_prompt = """
Ты выбираешь лучший фрагмент текста, в котором есть искомая информация. 
Дай только номер чанка (от 0 до 4) без дополнительного текста.

Тебе даются:
- описание того, какая именно информация нужна,
- несколько кандидатов-чанков.

Нужно выбрать лучший чанк и вернуть только его номер (от 0 до 4).
Ответь строго одним числом без дополнительного текста.
"""

        numbered_chunks = "\n\n".join(
            f"[{i}] {chunk}" for i, chunk in enumerate(top_chunks)
        )

        llm_question = f"""
Что нужно найти:
{info_to_find}

Кандидатные фрагменты:
{numbered_chunks}

Какой фрагмент наиболее полезен для ответа на вопрос?
Ответь только числом.
"""

        ans = run_llm_system_prompt(system_prompt, llm_question)

        match = re.search(r"\b([0-4])\b", ans)
        if match:
            chosen_idx = int(match.group(1))
        else:
            chosen_idx = 0

        if chosen_idx < 0 or chosen_idx >= len(top_chunks):
            chosen_idx = 0

        best_chunk = top_chunks[chosen_idx]
        wiki_best_chunks[query] = best_chunk

    return {**state, "wiki_best_chunks": wiki_best_chunks}


def node_compose_final_answer(state: AgentState) -> AgentState:
    """Формирует черновой финальный ответ на основе вопроса и ответов по отдельным запросам"""
    
    question = state["question"]
    wiki_best_chunks = state.get("wiki_best_chunks", {})

    system_prompt = """
Ты формируешь финальный ответ на исходный вопрос, опираясь на информацию из Википедии по каждому запросу.
Считай предоставленную информацию достоверной.
"""

    pieces: List[str] = [
        f"Вопрос: {question}",
        "Информация из Википедии:"
    ]

    for ans in wiki_best_chunks.values():
        pieces.append(f"- {ans}")

    context = "\n".join(pieces)
    final_answer = run_llm_system_prompt(system_prompt, context)

    return {**state, "final_answer": final_answer.strip()}


def node_self_check_answer(state: AgentState) -> AgentState:
    """Запрашивает у LLM самопроверку ответа по фактам из Википедии 
    и сохраняет флаг корректности и комментарий"""
    
    import json

    question = state["question"]
    wiki_best_chunks = state.get("wiki_best_chunks", {})
    final_answer = state.get("final_answer", "")

    original_final_answer = final_answer

    system_prompt = """
Ты модуль самопроверки ответа агента.

У тебя есть:
- исходный вопрос пользователя,
- информация, извлечённая из Википедии,
- черновой ответ агента.

Считай информацию, предоставленную из Википедии полностью достоверной.
Твоя задача проверить, содержится ли в черновом ответе ответ на поставленный вопрос, построенный на извлеченной информации.

Верни ТОЛЬКО JSON следующего вида:
{
  "ok": true,
  "feedback": "краткое описание, что не так или подтверждение корректности"
}
"""

    facts_lines = []
    for chunk in wiki_best_chunks.values():
        if not chunk.strip():
            continue
        facts_lines.append(chunk)

    facts_block = "\n".join(facts_lines) if facts_lines else "Фактов из Википедии нет."

    context = f"""
Вопрос пользователя:
{question}

Информация из Википедии:
{facts_block}

Черновой ответ агента:
{final_answer}
"""

    raw = run_llm_system_prompt(system_prompt, context)

    ok = True
    feedback = "Не удалось разобрать результат самопроверки, считаю ответ корректным."

    try:
        data = json.loads(raw)
        ok = bool(data.get("ok", True))
        fb = str(data.get("feedback", "")).strip()
        if fb:
            feedback = fb
    except Exception:
        pass

    new_state: AgentState = {
        **state,
        "original_final_answer": original_final_answer,
        "answer_ok": ok,
        "answer_feedback": feedback,
    }

    return new_state


def node_regenerate_answer(state: AgentState) -> AgentState:
    """Перегенерирует ответ с учётом замечаний самопроверки"""
    
    question = state["question"]
    wiki_best_chunks = state.get("wiki_best_chunks", {})
    prev_answer = state.get("final_answer", "")
    feedback = state.get("answer_feedback", "")

    system_prompt = """
Ты переписываешь ответ агента, исправляя ошибки.

У тебя есть:
- исходный вопрос,
- факты (фрагменты из Википедии),
- предыдущий ответ агента,
- замечания по этому ответу.

Сформируй НОВЫЙ окончательный ответ, который:
- согласуется с фактами,
- учитывает замечания,
- не повторяет явные ошибки исходного ответа.

Верни ТОЛЬКО текст нового ответа, без пояснений и без JSON.
"""
    facts_lines = []
    for q, chunk in wiki_best_chunks.items():
        if not chunk.strip():
            continue
        facts_lines.append(f"Запрос: {q}\nФрагмент: {chunk}")

    facts_block = "\n\n".join(facts_lines) if facts_lines else "Фактов из Википедии нет."

    context = f"""
Вопрос пользователя:
{question}

Факты (фрагменты из Википедии):
{facts_block}

Предыдущий ответ агента:
{prev_answer}

Замечания по предыдущему ответу:
{feedback}
"""

    new_answer = run_llm_system_prompt(system_prompt, context).strip()

    if not new_answer:
        # на всякий случай — если LLM сломалась, не теряем старый ответ
        new_answer = prev_answer

    new_state: AgentState = {
        **state,
        "final_answer": new_answer
    }

    return new_state


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("extract_queries", node_extract_search_queries)
    graph.add_node("fetch_wiki_pages", node_fetch_wiki_pages)
    graph.add_node("select_best_chunk", node_select_best_chunk)
    graph.add_node("compose_final_answer", node_compose_final_answer)
    graph.add_node("self_check_answer", node_self_check_answer)
    graph.add_node("regenerate_answer", node_regenerate_answer)

    graph.set_entry_point("extract_queries")
    graph.add_edge("extract_queries", "fetch_wiki_pages")
    graph.add_edge("fetch_wiki_pages", "select_best_chunk")
    graph.add_edge("select_best_chunk", "compose_final_answer")
    graph.add_edge("compose_final_answer", "self_check_answer")

    # условный переход после самопроверки
    graph.add_conditional_edges(
        "self_check_answer",
        lambda s: "ok" if s.get("answer_ok", True) else "retry",
        {
            "ok": END,
            "retry": "regenerate_answer",
        },
    )

    graph.add_edge("regenerate_answer", END)

    return graph.compile()

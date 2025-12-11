import httpx
from typing import List
from bs4 import BeautifulSoup
import re

WIKI_API_URL = "https://ru.wikipedia.org/w/api.php"

# без этого Википедия отдавала ошибку 403
HEADERS = {
    "User-Agent": "WikiLangGraphAgent/1.0 (https://example.com/; email@example.com)"
}

async def search_wiki(query: str, limit: int = 3) -> List[dict]:
    """
    Выполняет поиск по Википедии

    Параметры:
        query: Строка поиска
        limit: Максимальное число результатов

    Возвращает:
        Список поисковых документов (dict), каждый содержит title, snippet, pageid
        В случае ошибки возвращает пустой список
    """
    
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
    }

    try:
        async with httpx.AsyncClient(timeout=10, headers=HEADERS) as client:
            resp = await client.get(WIKI_API_URL, params=params)
            if resp.status_code != 200:
                print(f"[search_wiki] status={resp.status_code} query={query!r}")
                return []
            data = resp.json()
    except httpx.RequestError as e:
        print(f"[search_wiki] network error: {e} query={query!r}")
        return []
    except Exception as e:
        print(f"[search_wiki] unexpected error: {e} query={query!r}")
        return []

    return data.get("query", {}).get("search", [])


async def get_page_extract(pageid: int) -> str:
    """
    Получает краткую выжимку страницы Википедии по её pageid

    Забирает HTML через API, затем:
      1) извлекает инфобокс (если есть)
      2) извлекает лид-параграфы
      3) удаляет ссылки, номера примечаний, спецсимволы
      4) нормализует пробелы и пустые строки

    Параметры:
        pageid: Идентификатор страницы

    Возвращает:
        Очищенный текстовый фрагмент (инфобокс + лид)
    """
    
    async with httpx.AsyncClient(headers=HEADERS) as client:
        resp = await client.get(
            WIKI_API_URL,
            params={
                "action": "parse",
                "pageid": pageid,
                "prop": "text",
                "format": "json",
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()

    html = data["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")

    # 1) Инфобокс
    infobox_text = ""
    infobox = soup.find("table", {"class": "infobox"})
    if infobox:
        rows = infobox.find_all("tr")
        items = []
        for row in rows:
            cols = row.find_all(["th", "td"])
            if len(cols) == 2:
                key = cols[0].get_text(" ", strip=True)
                val = cols[1].get_text(" ", strip=True)
                if key and val:
                    items.append(f"{key}: {val}")
        infobox_text = "\n".join(items)

    # 2) Лид (до первого h2)
    content = soup.find("div", class_="mw-parser-output")
    lead_parts = []

    if content:
        for child in content.children:
            if getattr(child, "name", None) == "h2":
                break
            if child.name in ("p", "ul", "ol"):
                txt = child.get_text(" ", strip=True)
                if txt:
                    lead_parts.append(txt)

    lead_text = "\n".join(lead_parts)

    pieces = []
    if infobox_text:
        pieces.append(infobox_text)
    if lead_text:
        pieces.append(lead_text)

    result = "\n\n".join(pieces)

    # 3-4) Очистка от ссылок, номеров примечаний и мусора + нормализация
    result = re.sub(r"\[\s*\d+\s*\]", "", result)
    result = result.replace("↑", "")
    result = re.sub(r"https?://\S+", "", result)
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"\n\s*\n\s*\n+", "\n\n", result)

    return result.strip()

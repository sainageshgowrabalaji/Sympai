"""
Search MedlinePlus, the US National Library of Medicine's health encyclopedia.

This is SympAI's first tool. A tool has two halves.

1. The Python function, which does the real work (an HTTP call here).
2. The description the model reads, which is the docstring of the function
   below. The model never sees the code. It decides whether to call the tool
   from the name, the docstring and the argument types alone. So the docstring
   is really a prompt, and it deserves the same care.

MedlinePlus is free, needs no key, and asks callers to stay under
85 requests a minute.
"""

import html
import re
import xml.etree.ElementTree as ET

import httpx
from langchain_core.tools import tool

MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"

# Each topic summary is trimmed to this many characters. Three topics at 1200
# characters is roughly 900 tokens. Everything a tool returns goes back to the
# model on every later step, and Groq's free tier allows 8K tokens a minute,
# so tool output has to stay small.
SUMMARY_CHARS = 1200

_TAG = re.compile(r"<[^>]+>")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


def _clean(text: str | None) -> str:
    """MedlinePlus puts HTML inside its XML. Strip the tags, keep the words."""
    no_tags = _TAG.sub(" ", text or "")
    collapsed = " ".join(no_tags.split())
    return html.unescape(_SPACE_BEFORE_PUNCT.sub(r"\1", collapsed))


def parse_results(xml_bytes: bytes, max_chars: int = SUMMARY_CHARS) -> list[dict]:
    """Turn the MedlinePlus XML reply into a list of {title, url, summary}."""
    root = ET.fromstring(xml_bytes)
    topics = []
    for document in root.iter("document"):
        fields = {c.get("name"): _clean(c.text) for c in document.findall("content")}
        summary = fields.get("FullSummary", "")
        if len(summary) > max_chars:
            summary = summary[:max_chars].rsplit(" ", 1)[0] + " ..."
        topics.append(
            {
                "title": fields.get("title", ""),
                "url": document.get("url", ""),
                "summary": summary,
            }
        )
    return topics


@tool
def search_medlineplus(query: str) -> str:
    """Search MedlinePlus, the US National Library of Medicine's trusted health encyclopedia.

    Use this for plain-language facts about a symptom, condition, test, medicine
    or type of care. Pass a short medical search phrase, for example
    "chest pain", "strep throat" or "urgent care". Returns up to 3 topics,
    each with a title, a link to cite, and a summary.
    """
    try:
        response = httpx.get(
            MEDLINEPLUS_URL,
            params={"db": "healthTopics", "term": query, "retmax": 3},
            timeout=15,
        )
        response.raise_for_status()
        topics = parse_results(response.content)
    except (httpx.HTTPError, ET.ParseError) as error:
        # A tool should report failure as text rather than crash. The model
        # reads this and can retry, rephrase, or tell the person.
        return f"MedlinePlus could not be reached ({error}). Answer without it and say so."

    if not topics:
        # This used to say "try a simpler phrase", and in the visual snow test
        # the model obeyed three times in a row. A tool's words steer the agent.
        return (
            f"MedlinePlus has no page for '{query}'. Do not search for it again. "
            "Use the department reports, or say plainly that MedlinePlus has no page on it."
        )

    return "\n\n".join(f"{t['title']}\n{t['url']}\n{t['summary']}" for t in topics)

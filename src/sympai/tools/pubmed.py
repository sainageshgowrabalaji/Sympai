"""
Step 7. Search PubMed for recent reviews, part of the research center.

PubMed is the US National Library of Medicine's index of medical research.
We ask for reviews and systematic reviews from the last 5 years only, because
a review sums up many studies, which suits a person better than one paper.

Two calls, the NCBI E-utilities way
    esearch   finds the IDs of matching articles
    esummary  gets the title, journal and date for those IDs

Free and needs no key. NCBI asks for no more than 3 requests a second and a
"tool" name, so it knows who is calling. Only search words are sent, never the chart.
"""

import httpx
from langchain_core.tools import tool

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_RESULTS = 3


def parse_ids(data: dict) -> list[str]:
    return list((data.get("esearchresult") or {}).get("idlist") or [])


def parse_summaries(data: dict) -> list[dict]:
    result = data.get("result") or {}
    articles = []
    for uid in result.get("uids") or []:
        item = result.get(uid) or {}
        articles.append(
            {
                "title": (item.get("title") or "").strip(),
                "journal": item.get("source") or "",
                "year": (item.get("pubdate") or "")[:4],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            }
        )
    return articles


@tool
def search_pubmed(query: str) -> str:
    """Find recent research reviews on PubMed about a condition or symptom.

    Use this when the person asks what research says, or for a rare or
    long-lasting condition that MedlinePlus has no page for, such as visual
    snow syndrome. Pass a short medical phrase. Returns up to 3 reviews from
    the last 5 years, each with title, journal, year and link. These are
    research summaries, not advice for this person.
    """
    term = f'({query}) AND (review[pt] OR systematic[sb]) AND "last 5 years"[dp]'
    try:
        with httpx.Client(timeout=15) as client:
            found = client.get(
                f"{EUTILS}/esearch.fcgi",
                params={"db": "pubmed", "term": term, "retmax": MAX_RESULTS, "sort": "relevance",
                        "retmode": "json", "tool": "sympai"},
            )
            found.raise_for_status()
            ids = parse_ids(found.json())
            if not ids:
                return f"PubMed has no recent reviews on '{query}'. Do not search for it again."
            summaries = client.get(
                f"{EUTILS}/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json", "tool": "sympai"},
            )
            summaries.raise_for_status()
            articles = parse_summaries(summaries.json())
    except (httpx.HTTPError, ValueError) as error:
        return f"PubMed could not be reached ({error}). Answer without it and say so."
    lines = [f"{a['title']} ({a['journal']}, {a['year']})\n{a['url']}" for a in articles]
    return "Recent reviews. Research, not personal advice.\n\n" + "\n\n".join(lines)

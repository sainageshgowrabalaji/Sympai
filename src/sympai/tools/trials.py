"""
Step 7. Search ClinicalTrials.gov for studies recruiting now, part of the research center.

ClinicalTrials.gov is the US registry of clinical studies, run by the
National Library of Medicine. Its API v2 is free and needs no key.

Honesty matters here more than anywhere. A trial is a study, not a
treatment, and it may not help. Joining one is a decision to make with a
doctor. The tool says so in every result, so the coordinator cannot forget.
"""

import httpx
from langchain_core.tools import tool

API = "https://clinicaltrials.gov/api/v2/studies"
MAX_RESULTS = 3


def parse_studies(data: dict, location: str = "") -> list[dict]:
    wanted = location.strip().lower()
    studies = []
    for study in data.get("studies") or []:
        protocol = study.get("protocolSection") or {}
        ident = protocol.get("identificationModule") or {}
        status = protocol.get("statusModule") or {}
        design = protocol.get("designModule") or {}
        places = (protocol.get("contactsLocationsModule") or {}).get("locations") or []
        # Show the sites nearest in name to what the person asked for, first.
        places = sorted(places, key=lambda p: wanted not in " ".join(str(v) for v in p.values()).lower())
        nct = ident.get("nctId") or ""
        studies.append(
            {
                "title": ident.get("briefTitle") or "",
                "status": (status.get("overallStatus") or "").replace("_", " ").lower(),
                "kind": " ".join(
                    [(design.get("studyType") or "").lower(), ", ".join(p.replace("_", " ").lower() for p in design.get("phases") or [])]
                ).strip(),
                "where": "; ".join(
                    ", ".join(x for x in (p.get("facility"), p.get("city"), p.get("state")) if x) for p in places[:2]
                ),
                "url": f"https://clinicaltrials.gov/study/{nct}" if nct else "",
            }
        )
    return studies


@tool
def search_clinical_trials(condition: str, location: str = "") -> str:
    """Find clinical studies that are recruiting now, on ClinicalTrials.gov.

    Use this when the person asks about research they could join, or for a
    rare or long-lasting condition with no clear treatment. Pass the condition,
    and optionally a US state or city such as "New Jersey". Returns up to 3
    studies with title, type, where and a link. A trial is a study, not a treatment.
    """
    params = {"query.cond": condition, "filter.overallStatus": "RECRUITING", "pageSize": MAX_RESULTS, "format": "json"}
    if location:
        params["query.locn"] = location
    try:
        response = httpx.get(API, params=params, timeout=15)
        response.raise_for_status()
        studies = parse_studies(response.json(), location)
    except (httpx.HTTPError, ValueError) as error:
        return f"ClinicalTrials.gov could not be reached ({error}). Answer without it and say so."
    if not studies:
        where = f" near {location}" if location else ""
        return f"No recruiting studies found for '{condition}'{where}. Do not search for it again."
    lines = [f"{s['title']}\n{s['kind']}, {s['status']}\n{s['where'] or 'locations not listed'}\n{s['url']}" for s in studies]
    return (
        "Studies recruiting now. A trial is a study, not a treatment, and it may not help. "
        "Joining one is a decision to make with a doctor.\n\n" + "\n\n".join(lines)
    )

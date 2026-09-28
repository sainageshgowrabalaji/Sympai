"""The research center and patient services tools, tested offline against saved API shapes."""

import httpx
from langchain_core.messages import AIMessage

from sympai.tools import care_options, find_clinicians, search_clinical_trials, search_pubmed
from sympai.tools.pubmed import parse_ids, parse_summaries
from sympai.tools.services import maps_link, parse_clinicians
from sympai.tools.trials import parse_studies

ESEARCH = {"esearchresult": {"count": "2", "idlist": ["38000001", "37000002"]}}
ESUMMARY = {
    "result": {
        "uids": ["38000001", "37000002"],
        "38000001": {"uid": "38000001", "title": "Visual snow syndrome, a review.", "source": "J Neurol", "pubdate": "2024 Mar"},
        "37000002": {"uid": "37000002", "title": "Vestibular migraine update.", "source": "Headache", "pubdate": "2023"},
    }
}
TRIALS = {
    "studies": [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT05000001", "briefTitle": "A study of visual snow"},
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE2"]},
                "contactsLocationsModule": {
                    "locations": [
                        {"facility": "Clinic A", "city": "Boston", "state": "Massachusetts"},
                        {"facility": "Clinic B", "city": "Newark", "state": "New Jersey"},
                    ]
                },
            }
        }
    ]
}
NPI = {
    "result_count": 1,
    "results": [
        {
            "number": 1234567890,
            "basic": {"first_name": "JANE", "last_name": "DOE", "credential": "M.D."},
            "addresses": [
                {"address_purpose": "MAILING", "address_1": "PO BOX 1", "city": "X", "state": "NJ", "postal_code": "07102"},
                {"address_purpose": "LOCATION", "address_1": "10 MAIN ST", "city": "NEWARK", "state": "NJ",
                 "postal_code": "071021234", "telephone_number": "201-555-0100"},
            ],
            "taxonomies": [{"desc": "Otolaryngology", "primary": True}],
        }
    ],
}


def fake_get(responses):
    """Stand in for httpx.get, and for httpx.Client().get, returning saved JSON by URL."""

    def get(url, *args, **kwargs):
        for part, body in responses.items():
            if part in url:
                return httpx.Response(200, json=body, request=httpx.Request("GET", url))
        raise AssertionError(f"unexpected call to {url}")

    return get


def test_pubmed_parsing():
    assert parse_ids(ESEARCH) == ["38000001", "37000002"]
    first = parse_summaries(ESUMMARY)[0]
    assert first == {"title": "Visual snow syndrome, a review.", "journal": "J Neurol", "year": "2024",
                     "url": "https://pubmed.ncbi.nlm.nih.gov/38000001/"}


def test_pubmed_tool_labels_research_as_research(monkeypatch):
    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **kw: fake_get({"esearch": ESEARCH, "esummary": ESUMMARY})(url))
    text = search_pubmed.invoke({"query": "visual snow"})
    assert text.startswith("Recent reviews. Research, not personal advice.")
    assert "https://pubmed.ncbi.nlm.nih.gov/38000001/" in text


def test_trials_put_the_requested_state_first():
    [study] = parse_studies(TRIALS, "New Jersey")
    assert study["where"].startswith("Clinic B, Newark, New Jersey")
    assert study["url"] == "https://clinicaltrials.gov/study/NCT05000001"
    assert study["kind"] == "interventional phase2"


def test_trials_tool_always_says_a_trial_is_not_a_treatment(monkeypatch):
    monkeypatch.setattr(httpx, "get", fake_get({"clinicaltrials.gov": TRIALS}))
    text = search_clinical_trials.invoke({"condition": "visual snow", "location": "New Jersey"})
    assert "A trial is a study, not a treatment" in text


def test_npi_uses_the_practice_address_not_the_mailing_one():
    [doc] = parse_clinicians(NPI)
    assert doc["name"] == "Jane Doe, MD"
    assert doc["address"] == "10 Main St, Newark, NJ 07102"
    assert doc["phone"] == "201-555-0100"
    assert doc["map"].startswith("https://www.google.com/maps/search/?api=1&query=Jane+Doe")


def test_find_clinicians_says_to_check_insurance(monkeypatch):
    monkeypatch.setattr(httpx, "get", fake_get({"npiregistry": NPI}))
    text = find_clinicians.invoke({"specialty": "ENT", "zip_code": "07102"})
    assert "Jane Doe, MD" in text and "check your insurer" in text


def test_a_bad_zip_is_caught_before_any_call():
    assert "not a 5-digit ZIP code" in find_clinicians.invoke({"specialty": "ENT", "zip_code": "73"})
    assert "not a 5-digit ZIP code" in care_options.invoke({"zip_code": "abc"})


def test_care_options_are_links_not_invented_places():
    text = care_options.invoke({"zip_code": "07102", "need": "urgent care"})
    assert maps_link("urgent care near 07102") in text
    assert "findahealthcenter.hrsa.gov" in text and "988" in text


def test_the_chat_page_gets_clinicians_as_data_not_text(monkeypatch):
    """The model reads the text. The page draws cards from the artifact, and never parses sentences."""
    from sympai.graph import services

    monkeypatch.setattr(httpx, "get", fake_get({"npiregistry": NPI}))
    last = AIMessage("", tool_calls=[{"name": "find_clinicians", "args": {"specialty": "ENT", "zip_code": "07102"}, "id": "c1", "type": "tool_call"}])
    [message] = services({"messages": [last]})["messages"]
    assert "Jane Doe, MD" in message.content  # what the model sees, unchanged
    assert message.artifact["kind"] == "clinicians"
    assert message.artifact["items"][0]["phone"] == "201-555-0100"


def test_care_options_links_come_as_data_too():
    from sympai.graph import services

    last = AIMessage("", tool_calls=[{"name": "care_options", "args": {"zip_code": "07102"}, "id": "c2", "type": "tool_call"}])
    [message] = services({"messages": [last]})["messages"]
    assert [item["label"] for item in message.artifact["items"]][:2] == ["Urgent care near 07102", "Pharmacies near 07102"]

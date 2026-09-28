"""
Step 8. Patient services. Real places, from real sources, instead of invented ones.

Early on, SympAI made up a clinic, an address and a train route. The rule
since then has been "never name a place". These two tools are the fix that
lets it name places safely, because every name, address and phone number
comes from a public source, not from the model.

find_clinicians   the NPI Registry, the US government's list of every licensed
                  clinician, run by CMS. Free, no key.
care_options      map searches and national services for urgent care,
                  pharmacies, low-cost clinics, nurse lines and crisis numbers.

What neither can tell you is whether a clinician takes your insurance or new
patients, so every result says to check with the insurer or the office.
Ratings are one tap away on the Google Maps link.
"""

import re
from urllib.parse import quote_plus

import httpx
from langchain_core.tools import tool

NPI_API = "https://npiregistry.cms.hhs.gov/api/"
MAX_RESULTS = 5

# Plain words the coordinator might use, mapped to the NPI Registry's taxonomy names.
SPECIALTIES = {
    "primary care": "Family Medicine",
    "family medicine": "Family Medicine",
    "internal medicine": "Internal Medicine",
    "ent": "Otolaryngology",
    "otolaryngology": "Otolaryngology",
    "neurology": "Neurology",
    "cardiology": "Cardiovascular Disease",
    "heart": "Cardiovascular Disease",
    "lungs": "Pulmonary Disease",
    "pulmonology": "Pulmonary Disease",
    "dermatology": "Dermatology",
    "skin": "Dermatology",
    "eyes": "Ophthalmology",
    "ophthalmology": "Ophthalmology",
    "optometry": "Optometrist",
    "orthopedics": "Orthopaedic Surgery",
    "bones and muscles": "Orthopaedic Surgery",
    "physical therapy": "Physical Therapist",
    "psychiatry": "Psychiatry & Neurology, Psychiatry",
    "mental health": "Psychiatry & Neurology, Psychiatry",
    "therapy": "Counselor",
    "audiology": "Audiologist",
    "gastroenterology": "Gastroenterology",
    "womens health": "Obstetrics & Gynecology",
    "pediatrics": "Pediatrics",
}

HOW_TO_CHECK = (
    "The NPI Registry lists licensed clinicians. It does not say whether they take your insurance or new "
    "patients, so check your insurer's find-a-doctor tool or call the office first. "
    "Ratings and directions are on each map link."
)


def maps_link(query: str) -> str:
    """A Google Maps search link. Official Maps URL format, needs no key."""
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"


def parse_clinicians(data: dict) -> list[dict]:
    people = []
    for entry in data.get("results") or []:
        basic = entry.get("basic") or {}
        name = " ".join(x for x in (basic.get("first_name"), basic.get("last_name")) if x).title()
        name = basic.get("organization_name") or name
        credential = (basic.get("credential") or "").replace(".", "")
        place = next((a for a in entry.get("addresses") or [] if a.get("address_purpose") == "LOCATION"), None)
        if not place or not name:
            continue
        taxonomy = next((t for t in entry.get("taxonomies") or [] if t.get("primary")), None) or {}
        zip5 = str(place.get("postal_code") or "")[:5]
        address = f"{(place.get('address_1') or '').title()}, {(place.get('city') or '').title()}, {place.get('state', '')} {zip5}"
        people.append(
            {
                "name": f"{name}, {credential}" if credential else name,
                "specialty": taxonomy.get("desc") or "",
                "address": address,
                "phone": place.get("telephone_number") or "",
                "map": maps_link(f"{name} {address}"),
            }
        )
    return people


# content_and_artifact means each tool returns two things. The text is what the
# model reads. The artifact is the same result as data, which never goes to the
# model, and SympAI's own chat page draws it as cards with Call and Map buttons.
@tool(response_format="content_and_artifact")
def find_clinicians(specialty: str, zip_code: str) -> tuple[str, dict | None]:
    """Find licensed clinicians of one kind near a US ZIP code, from the official NPI Registry.

    Use this when the person needs a specific kind of doctor or therapist and
    you know their ZIP code. Specialty is plain words such as primary care,
    ENT, neurology, cardiology, dermatology, eyes, orthopedics, physical
    therapy, mental health or audiology. Returns up to 5 names with address,
    phone and a map link. Only ever name clinicians that this tool returned.
    """
    zip5 = re.sub(r"\D", "", zip_code)[:5]
    if len(zip5) != 5:
        return "That is not a 5-digit ZIP code. Ask the person for their ZIP code.", None
    taxonomy = SPECIALTIES.get(specialty.strip().lower(), specialty)
    params = {"version": "2.1", "taxonomy_description": taxonomy, "postal_code": zip5,
              "address_purpose": "LOCATION", "limit": MAX_RESULTS}
    try:
        response = httpx.get(NPI_API, params=params, timeout=15)
        response.raise_for_status()
        people = parse_clinicians(response.json())
    except (httpx.HTTPError, ValueError) as error:
        return f"The NPI Registry could not be reached ({error}). Use care_options for a map search instead.", None
    if not people:
        search = maps_link(f"{specialty} near {zip5}")
        text = (
            f"The NPI Registry has no {taxonomy} listed in ZIP {zip5}. Nearby ZIP codes may have some. "
            f"A map search also works: {search}"
        )
        return text, {"kind": "links", "items": [{"label": f"{specialty.capitalize()} near {zip5}", "url": search}]}
    lines = [f"{p['name']}. {p['specialty']}\n{p['address']}\nPhone {p['phone']}\nMap {p['map']}" for p in people]
    return "\n\n".join(lines) + "\n\n" + HOW_TO_CHECK, {"kind": "clinicians", "items": people}


@tool(response_format="content_and_artifact")
def care_options(zip_code: str, need: str = "urgent care") -> tuple[str, dict | None]:
    """Where to get everyday care near a US ZIP code, with real links instead of guesses.

    Use this for urgent care, a pharmacy, a walk-in clinic, low-cost or
    sliding-scale care, or telehealth. Need is plain words such as urgent
    care, pharmacy or low-cost clinic. Returns map searches and national
    services. It never makes up a place.
    """
    zip5 = re.sub(r"\D", "", zip_code)[:5]
    if len(zip5) != 5:
        return "That is not a 5-digit ZIP code. Ask the person for their ZIP code.", None
    links = [
        {"label": f"{need.capitalize()} near {zip5}", "url": maps_link(f"{need} near {zip5}")},
        {"label": f"Pharmacies near {zip5}", "url": maps_link(f"pharmacy near {zip5}")},
        {"label": "Low-cost community health centers", "url": "https://findahealthcenter.hrsa.gov"},
    ]
    return "\n".join(
        [
            f"{need.capitalize()} near {zip5}, with ratings, hours and directions: {links[0]['url']}",
            f"Pharmacies near {zip5}: {links[1]['url']}",
            "Low-cost care. Community health centers charge on a sliding scale by income, even without insurance. "
            "Find one at https://findahealthcenter.hrsa.gov",
            "Telehealth and nurse lines. Many insurance plans include telehealth visits, and many cards list a "
            "24-hour nurse line on the back. Check the insurer's app or website.",
            "Emergencies call 911. Thoughts of suicide or self-harm call or text 988. Poison Control 1-800-222-1222.",
        ]
    ), {"kind": "links", "items": links}


SERVICE_TOOLS = [find_clinicians, care_options]

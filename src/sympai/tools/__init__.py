"""Tools are plain Python functions the model is allowed to ask for."""

from sympai.tools.medlineplus import search_medlineplus
from sympai.tools.pubmed import search_pubmed
from sympai.tools.services import SERVICE_TOOLS, care_options, find_clinicians
from sympai.tools.trials import search_clinical_trials

RESEARCH_TOOLS = [search_medlineplus, search_pubmed, search_clinical_trials]

__all__ = [
    "RESEARCH_TOOLS",
    "SERVICE_TOOLS",
    "care_options",
    "find_clinicians",
    "search_clinical_trials",
    "search_medlineplus",
    "search_pubmed",
]

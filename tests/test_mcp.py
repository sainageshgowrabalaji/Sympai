"""The MCP server offers the same tools SympAI uses, and they work through MCP."""

import anyio

from sympai.mcp_server import server


def test_the_server_lists_all_five_tools():
    tools = anyio.run(server.list_tools)
    assert {t.name for t in tools} == {"medlineplus", "pubmed_reviews", "clinical_trials", "clinicians", "where_to_get_care"}
    assert all(t.description for t in tools)


def test_a_tool_can_be_called_through_mcp():
    result = anyio.run(server.call_tool, "where_to_get_care", {"zip_code": "07102", "need": "pharmacy"})
    text = str(result)
    assert "Pharmacy near 07102" in text and "findahealthcenter.hrsa.gov" in text

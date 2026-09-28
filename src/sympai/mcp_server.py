"""
Step 7. The research center as an MCP server.

MCP, the Model Context Protocol, is a standard way to offer tools to any AI
agent. Think of it as an OpenAPI contract, but for AI tools. SympAI's own
graph calls these tools directly, in the same process, which is fastest.
Publishing them over MCP means other agents can use them too, for example
Claude in the desktop app, without knowing anything about SympAI's code.

Run it on its own to check it starts
    uv run python -m sympai.mcp_server

To use it from the Claude desktop app, add this to claude_desktop_config.json
    "sympai-research": {
      "command": "uv",
      "args": ["--directory", "/full/path/to/sympai", "run", "python", "-m", "sympai.mcp_server"]
    }
"""

from mcp.server.mcpserver import MCPServer

from sympai.tools import care_options, find_clinicians, search_clinical_trials, search_medlineplus, search_pubmed

server = MCPServer(
    name="sympai-research",
    instructions=(
        "Public US health information. MedlinePlus pages, PubMed reviews, recruiting clinical trials, "
        "licensed clinicians from the NPI Registry, and where to get care. Research is not personal advice."
    ),
)


# Each MCP tool is a thin wrapper over the same LangChain tool SympAI uses,
# so there is one implementation and two ways in.
@server.tool(description=search_medlineplus.description)
def medlineplus(query: str) -> str:
    return search_medlineplus.invoke({"query": query})


@server.tool(description=search_pubmed.description)
def pubmed_reviews(query: str) -> str:
    return search_pubmed.invoke({"query": query})


@server.tool(description=search_clinical_trials.description)
def clinical_trials(condition: str, location: str = "") -> str:
    return search_clinical_trials.invoke({"condition": condition, "location": location})


@server.tool(description=find_clinicians.description)
def clinicians(specialty: str, zip_code: str) -> str:
    return find_clinicians.invoke({"specialty": specialty, "zip_code": zip_code})


@server.tool(description=care_options.description)
def where_to_get_care(zip_code: str, need: str = "urgent care") -> str:
    return care_options.invoke({"zip_code": zip_code, "need": need})


if __name__ == "__main__":
    server.run()  # stdio, the transport desktop apps use

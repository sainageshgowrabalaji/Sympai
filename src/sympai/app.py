"""The graph that LangGraph Studio loads. langgraph.json points here.

The Studio server brings its own checkpointer, so none is passed in.
"""

from dotenv import load_dotenv

from sympai.graph import build_graph

load_dotenv()
graph = build_graph()

"""
Step 2. An agent loop, written by hand.

An agent is a model in a loop with tools. That is the whole idea.

    1. Send the conversation to the model, along with the list of tools.
    2. The model either answers, or replies "please run this tool with
       these arguments". That reply is called a tool call.
    3. If it asked for a tool, WE run it (the model cannot run code) and
       add the result to the conversation as a ToolMessage.
    4. Go back to 1. Stop when the model answers without asking for a tool.

LangGraph, which we use from step 3, is this same loop with state,
branching, memory and streaming built in. Writing it once by hand is how
you understand what LangGraph is doing for you.

Run it
    uv run python scripts/agent_by_hand.py "I have a sore throat and fever since yesterday"
"""

import re
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from sympai.tools import search_medlineplus

SYSTEM = """You are SympAI, a careful US health navigator.
Before you explain a symptom or condition, look it up with search_medlineplus.
Say what it may be, and never say a diagnosis is confirmed.
If anything sounds like an emergency, tell the person to call 911 first.
Say which kind of care fits, such as self-care, a pharmacist, primary care, urgent care or the ER.
End with a Sources line that lists the plain MedlinePlus URLs you used.
Keep the answer under 150 words."""

TOOLS = [search_medlineplus]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# gpt-oss was trained with its own citation marks like 【2†L2-L9】. They point
# at nothing in our app. A prompt can ask the model to stop, but a habit from
# training can slip through, so we also remove them in code. Rule of thumb:
# anything that must ALWAYS hold gets enforced in code, not only in a prompt.
_TRAINING_CITATIONS = re.compile(r"\s*【[^】]*】")


def clean_answer(text: str) -> str:
    return _TRAINING_CITATIONS.sub("", text).strip()


# A guard rail for the loop itself. Every step resends the whole
# conversation, so a model that keeps asking for tools burns tokens fast.
MAX_STEPS = 4


def run(brain, question: str) -> str:
    """The agent loop. `brain` is a model that already knows about TOOLS."""
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]

    for step in range(1, MAX_STEPS + 1):
        reply = brain.invoke(messages)
        messages.append(reply)  # the model's turn becomes part of the history

        usage = reply.usage_metadata or {}
        print(
            f"[step {step}] sent {usage.get('input_tokens')} tokens, "
            f"got {usage.get('output_tokens')} back"
        )

        # No tool calls means the model is done and this is the answer.
        if not reply.tool_calls:
            return clean_answer(reply.content)

        for call in reply.tool_calls:
            print(f"[step {step}] model asked for {call['name']}({call['args']})")
            tool = TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                result = f"There is no tool called {call['name']}. Use only the tools you were given."
            else:
                result = tool.invoke(call["args"])
            print(f"[step {step}] tool returned {len(result)} characters")

            # tool_call_id ties this result to the request it answers, because
            # a model can ask for several tools in one turn.
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    return f"Stopped after {MAX_STEPS} steps without a final answer."


if __name__ == "__main__":
    load_dotenv()
    question = " ".join(sys.argv[1:]) or input("What is going on? ")

    # bind_tools sends each tool's name, docstring and argument types to the
    # model with every request. The model never sees the Python code itself.
    brain = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        reasoning_effort="medium",
    ).bind_tools(TOOLS)

    answer = run(brain, question)
    print()
    print(answer)

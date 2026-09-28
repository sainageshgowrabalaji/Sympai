"""
Step 1. One question to a model in the cloud.

This is the smallest useful AI program. It loads your key from .env,
sends one conversation to gpt-oss-120b on Groq, and prints the answer.
Everything SympAI does later is built on top of this one call.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Reads GROQ_API_KEY from the .env file in this folder into the environment.
# ChatGroq finds it there on its own, so the key never appears in code.
load_dotenv()

# The model is the "brain". temperature=0 means the same question gets
# the same answer, which is what you want while testing.
brain = ChatGroq(model="openai/gpt-oss-120b", temperature=0, reasoning_effort="high")
# A conversation is a list of messages. The system message sets the rules,
# the human message is what the person typed.
conversation = [
    ("system", "You are a careful US health navigator. Answer in two short sentences."),
    ("human", "What is the difference between urgent care and an emergency room?"),
]

reply = brain.invoke(conversation)

print(reply.content)
print()
print("Tokens used:", reply.usage_metadata)
print()
print("What it was thinking:", reply.additional_kwargs.get("reasoning_content"))

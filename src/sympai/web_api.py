"""
Step 12. Three small routes for SympAI's own chat page, next to LangGraph's API.

    GET    /sympai/me           who is signed in, and what SympAI remembers about them
    DELETE /sympai/me/chart     forget all of it
    POST   /sympai/transcribe   a voice recording in, the words and their language out

LangGraph lets a project mount its own Starlette app beside the built-in
routes, like adding a @RestController to an existing Spring Boot service.
langgraph.json turns on enable_custom_route_auth, so every request here
passes through the same key check in auth.py first. By the time a handler
runs, request.user is the signed-in friend, and each route only ever touches
that friend's own chart.

The chart store and Whisper are blocking calls, so they run on a worker thread.
The server has one event loop, and blocking it would pause everyone.
"""

import asyncio

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from sympai import chart as charts
from sympai.voice import transcribe

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # about 10 minutes of compressed speech
AUDIO_TYPES = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4", "audio/mpeg": "mp3", "audio/wav": "wav",
               "audio/x-m4a": "m4a", "audio/m4a": "m4a"}

_store = charts.LazyStore()


def signed_in(request: Request) -> str | None:
    user = request.scope.get("user")
    identity = getattr(user, "identity", None)
    return identity if identity and getattr(user, "is_authenticated", True) else None


def _refuse() -> JSONResponse:
    return JSONResponse({"detail": "Paste your SympAI key first."}, status_code=401)


def remembered(chart: charts.Chart) -> list[dict]:
    """The chart as labeled groups for the memory panel, leaving out what is empty."""
    groups = [
        ("Age", [chart.age] if chart.age else []),
        ("Long-term conditions", chart.conditions),
        ("Medicines", chart.medicines),
        ("Allergies", chart.allergies),
        ("Insurance", [chart.insurance] if chart.insurance else []),
        ("ZIP code", [chart.zip_code] if chart.zip_code else []),
        ("Primary care doctor", [] if chart.has_primary_care is None else ["Yes" if chart.has_primary_care else "Not yet"]),
        ("Health history", chart.history),
        ("What SympAI advised", chart.advice),
    ]
    return [{"label": label, "values": values} for label, values in groups if values]


async def me(request: Request) -> JSONResponse:
    person = signed_in(request)
    if not person:
        return _refuse()
    chart = await asyncio.to_thread(charts.load, _store, person)
    return JSONResponse({"name": person, "remembers": remembered(chart)})


async def forget_me(request: Request) -> JSONResponse:
    person = signed_in(request)
    if not person:
        return _refuse()
    await asyncio.to_thread(charts.forget, _store, person)
    return JSONResponse({"forgotten": True})


async def voice(request: Request) -> JSONResponse:
    if not signed_in(request):
        return _refuse()
    audio = await request.body()
    if not audio:
        return JSONResponse({"detail": "The recording was empty."}, status_code=400)
    if len(audio) > MAX_AUDIO_BYTES:
        return JSONResponse({"detail": "That recording is too long. Try a shorter message."}, status_code=413)
    kind = request.headers.get("content-type", "audio/webm").split(";")[0].strip().lower()
    extension = AUDIO_TYPES.get(kind, "webm")
    try:
        text, language = await asyncio.to_thread(transcribe, audio, None, f"speech.{extension}")
    except Exception as error:  # Groq down or rate limited. The person can still type.
        return JSONResponse({"detail": f"Could not hear that ({type(error).__name__}). Please type it instead."},
                            status_code=502)
    return JSONResponse({"text": text, "language": language})


app = Starlette(
    routes=[
        Route("/sympai/me", me, methods=["GET"]),
        Route("/sympai/me/chart", forget_me, methods=["DELETE"]),
        Route("/sympai/transcribe", voice, methods=["POST"]),
    ]
)

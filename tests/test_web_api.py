"""
Step 12. The chat page's routes, tested with Starlette's TestClient, like MockMvc
in Spring. A small middleware stands in for the server's key check and sets
request.user, the way the real auth middleware does.
"""

from types import SimpleNamespace

from langgraph.store.memory import InMemoryStore
from starlette.middleware import Middleware
from starlette.testclient import TestClient

from sympai import chart as charts
from sympai import web_api
from sympai.chart import EMPTY


class SignedInAs:
    def __init__(self, app, name):
        self.app, self.name = app, name

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.name:
            scope["user"] = SimpleNamespace(identity=self.name, is_authenticated=True)
        await self.app(scope, receive, send)


def client_for(name, monkeypatch, store=None):
    store = store or InMemoryStore()
    monkeypatch.setattr(web_api, "_store", store)
    app = web_api.Starlette(routes=web_api.app.routes, middleware=[Middleware(SignedInAs, name=name)])
    return TestClient(app), store


def test_a_friend_sees_only_what_is_remembered_about_them(monkeypatch):
    client, store = client_for("asha", monkeypatch)
    charts.save(store, "asha", EMPTY.model_copy(update={"age": "34", "zip_code": "07102"}))
    charts.save(store, "me", EMPTY.model_copy(update={"insurance": "Aetna Open Access"}))

    body = client.get("/sympai/me").json()
    assert body["name"] == "asha"
    assert body["remembers"] == [{"label": "Age", "values": ["34"]}, {"label": "ZIP code", "values": ["07102"]}]
    assert "Aetna" not in str(body)  # the owner's chart never shows up


def test_forget_me_deletes_only_my_chart(monkeypatch):
    client, store = client_for("asha", monkeypatch)
    charts.save(store, "asha", EMPTY.model_copy(update={"age": "34"}))
    charts.save(store, "me", EMPTY.model_copy(update={"age": "40"}))

    assert client.delete("/sympai/me/chart").json() == {"forgotten": True}
    assert charts.load(store, "asha") == EMPTY
    assert charts.load(store, "me").age == "40"


def test_nobody_signed_in_gets_nothing(monkeypatch):
    client, _ = client_for(None, monkeypatch)
    assert client.get("/sympai/me").status_code == 401
    assert client.post("/sympai/transcribe", content=b"audio").status_code == 401


def test_a_browser_recording_is_sent_to_whisper_with_its_real_format(monkeypatch):
    heard = {}

    def fake_transcribe(audio, client, filename):
        heard.update(audio=audio, filename=filename)
        return "నాకు తలనొప్పి ఉంది", "telugu"

    monkeypatch.setattr(web_api, "transcribe", fake_transcribe)
    client, _ = client_for("asha", monkeypatch)
    response = client.post("/sympai/transcribe", content=b"webm-bytes", headers={"content-type": "audio/webm;codecs=opus"})
    assert response.json() == {"text": "నాకు తలనొప్పి ఉంది", "language": "telugu"}
    assert heard == {"audio": b"webm-bytes", "filename": "speech.webm"}


def test_whisper_failing_asks_the_person_to_type(monkeypatch):
    def broken(audio, client, filename):
        raise ConnectionError("Groq did not answer")

    monkeypatch.setattr(web_api, "transcribe", broken)
    client, _ = client_for("asha", monkeypatch)
    response = client.post("/sympai/transcribe", content=b"x", headers={"content-type": "audio/mp4"})
    assert response.status_code == 502 and "type it" in response.json()["detail"]

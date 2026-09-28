"""
Step 11. The friends login, tested without a server. Each handler is a plain
async function, so the tests call it directly, the same way a unit test calls
a Spring Security filter without starting Tomcat.
"""

import asyncio
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph_sdk import Auth

from sympai import auth
from sympai.chart import person_of

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def keys(tmp_path, monkeypatch):
    path = tmp_path / "friends.json"
    monkeypatch.setenv("SYMPAI_FRIENDS_FILE", str(path))
    auth._messages_today.clear()
    return path


def run(coroutine):
    return asyncio.run(coroutine)


def ctx(identity, resource, action):
    return SimpleNamespace(user=SimpleNamespace(identity=identity), resource=resource, action=action, permissions=[])


def test_a_key_signs_in_as_its_friend(keys):
    key = auth.add_friend("Asha")
    assert key.startswith("sympai_") and len(key) > 30
    user = run(auth.authenticate({b"x-api-key": key.encode()}))
    assert user["identity"] == "asha"


def test_the_key_itself_is_never_saved(keys):
    key = auth.add_friend("asha")
    saved = keys.read_text()
    assert key not in saved and auth.hash_key(key) in saved
    assert stat.S_IMODE(keys.stat().st_mode) == 0o600


@pytest.mark.parametrize("header", [{}, {b"x-api-key": b""}, {b"x-api-key": b"sympai_wrong"}, {b"x-api-key": b"lsv2_langsmith"}])
def test_no_key_or_a_wrong_key_is_turned_away(keys, header):
    auth.add_friend("asha")
    with pytest.raises(Auth.exceptions.HTTPException) as refused:
        run(auth.authenticate(header))
    assert refused.value.status_code == 401


def test_a_removed_friend_is_turned_away_on_the_next_message(keys):
    key = auth.add_friend("asha")
    assert auth.remove_friend("asha")
    with pytest.raises(Auth.exceptions.HTTPException):
        run(auth.authenticate({b"x-api-key": key.encode()}))


def test_names_are_kept_simple(keys):
    for bad in ["", "Asha Rao", "../me", "eval-1", "a" * 40]:
        with pytest.raises(ValueError):
            auth.add_friend(bad)


def test_each_friend_only_sees_their_own_conversations(keys):
    new_thread = {"metadata": {}}
    only_mine = run(auth.own_threads_only(ctx("asha", "threads", "create"), new_thread))
    assert new_thread["metadata"]["owner"] == "asha"
    assert only_mine == {"owner": "asha"}
    # A search is filtered the same way, whatever the friend asked for.
    assert run(auth.own_threads_only(ctx("ravi", "threads", "search"), {"metadata": {"owner": "asha"}})) == {"owner": "ravi"}


def test_a_friend_cannot_claim_someone_elses_thread(keys):
    # They try to stamp the thread with another owner. The handler overwrites it.
    sneaky = {"metadata": {"owner": "me"}}
    run(auth.own_threads_only(ctx("asha", "threads", "create"), sneaky))
    assert sneaky["metadata"]["owner"] == "asha"


def test_friends_can_use_the_assistant_but_not_change_it(keys):
    assert run(auth.read_the_assistant(ctx("asha", "assistants", "read"), {})) is True
    handler = auth.auth._handlers.get(("assistants", "update")) or auth.auth._global_handlers
    assert handler == [auth.deny_everything_else]
    with pytest.raises(Auth.exceptions.HTTPException) as refused:
        run(auth.deny_everything_else(ctx("asha", "assistants", "update"), {}))
    assert refused.value.status_code == 403


def test_a_daily_cap_protects_the_shared_allowance(keys, monkeypatch):
    monkeypatch.setattr(auth, "DAILY_MESSAGES", 2)
    auth.count_message("asha", "2026-09-27")
    auth.count_message("asha", "2026-09-27")
    with pytest.raises(Auth.exceptions.HTTPException) as refused:
        auth.count_message("asha", "2026-09-27")
    assert refused.value.status_code == 429 and "911" in refused.value.detail
    auth.count_message("asha", "2026-09-28")  # a new day
    for _ in range(5):
        auth.count_message("me")  # the owner has no cap


def test_the_signed_in_name_picks_the_chart_not_what_the_request_says():
    # A friend sends person="me" hoping to read the owner's chart. The server-set name wins.
    config = {"configurable": {"person": "me", "langgraph_auth_user_id": "asha"}}
    assert person_of(config) == "asha"
    assert person_of({"configurable": {"person": "mom"}}) == "mom"  # the terminal chat, no sign-in


def test_the_server_config_turns_login_on_and_the_studio_bypass_off():
    """A guard against a quiet mistake. In langgraph dev, a request with the
    header x-auth-scheme: langsmith is treated as LangGraph Studio and skips
    authenticate entirely. Through a public tunnel, anyone could send it. This
    test fails if that door is ever left open."""
    config = json.loads((ROOT / "langgraph.json").read_text())
    assert config["auth"]["path"] == "./src/sympai/auth.py:auth"
    assert config["auth"]["disable_studio_auth"] is True
    # The chat page's own routes, voice and memory, must sit behind the same key check.
    assert config["http"]["enable_custom_route_auth"] is True


def test_friends_json_is_never_committed():
    assert "data/" in (ROOT / ".gitignore").read_text().split()

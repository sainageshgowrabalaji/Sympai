"""
Step 11. Friends. A key for each person, and each person sees only their own data.

The chat screen at agentchat.vercel.app sends whatever is typed into its
"LangSmith API Key" box as the x-api-key header on every request. For SympAI
that box holds a key made by scripts/friends.py, one per friend.

    authenticate    runs on every request, like a Spring Security filter. It
                    hashes the key, finds whose it is, and says who is asking.
                    The server then puts that name in the run's config as
                    langgraph_auth_user_id, and chart.person_of reads it, so
                    each friend gets their own chart.
    threads         stamps each conversation with its owner and filters every
                    read, search and run to the asker's own. Think of it as a
                    WHERE owner = ? added to every query.
    assistants      friends may read the SympAI assistant, never change it.
    everything else denied. LangGraph's custom auth fails open, which means
                    a request with no matching handler is let through. So the
                    base rule here is a catch-all that says no.

Keys are never stored. data/friends.json keeps only a SHA-256 hash of each,
the same way a database keeps password hashes, so the file alone cannot be
used to sign in. data/ is in .gitignore and never leaves the laptop.
"""

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import date
from pathlib import Path

from langgraph_sdk import Auth

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
KEY_PREFIX = "sympai_"
NAME_RULE = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")
# Groq's free allowance is shared by everyone. A cap per friend per day stops
# one long session from using it all. "me" is the owner and has no cap.
DAILY_MESSAGES = int(os.getenv("SYMPAI_DAILY_MESSAGES", "30"))
OWNER = "me"

auth = Auth()


# ---------------------------------------------------------------- the key file


def friends_file() -> Path:
    return Path(os.getenv("SYMPAI_FRIENDS_FILE") or DATA_DIR / "friends.json")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load_friends(path: Path | None = None) -> dict:
    path = path or friends_file()
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("friends", {})


def save_friends(friends: dict, path: Path | None = None) -> None:
    path = path or friends_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"friends": friends}, indent=2))
    path.chmod(0o600)  # readable by you only, like ~/.ssh keys


def add_friend(name: str, path: Path | None = None) -> str:
    """Make a new key for `name` and return it. It is shown once and never stored."""
    name = name.strip().lower()
    if not NAME_RULE.match(name) or name.startswith("eval-"):
        raise ValueError("Use a short name with lowercase letters, numbers and hyphens, like asha or ravi-k.")
    key = KEY_PREFIX + secrets.token_urlsafe(24)  # 192 random bits, the same strength as a strong API key
    friends = load_friends(path)
    friends[name] = {"key_sha256": hash_key(key), "added": date.today().isoformat()}
    save_friends(friends, path)
    return key


def remove_friend(name: str, path: Path | None = None) -> bool:
    friends = load_friends(path)
    if friends.pop(name.strip().lower(), None) is None:
        return False
    save_friends(friends, path)
    return True


def who_has(key: str | None, path: Path | None = None) -> str | None:
    """The friend this key belongs to, or None."""
    if not key or not key.startswith(KEY_PREFIX):
        return None
    wanted = hash_key(key)
    found = None
    for name, entry in load_friends(path).items():
        # compare_digest takes the same time whether the first or last character
        # differs, so response times leak nothing about a real key's hash.
        if hmac.compare_digest(entry.get("key_sha256", ""), wanted):
            found = name
    return found


# ---------------------------------------------------------------- who is asking


def _unauthorized() -> Auth.exceptions.HTTPException:
    return Auth.exceptions.HTTPException(status_code=401, detail="Paste your SympAI key into the API key box.")


@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    raw = headers.get(b"x-api-key") or headers.get("x-api-key")
    key = raw.decode() if isinstance(raw, bytes) else raw
    # The file is read in a worker thread. The server runs on one event loop,
    # and a blocking file read there would pause every other request.
    name = await asyncio.to_thread(who_has, key)
    if name is None:
        raise _unauthorized()
    return {"identity": name, "is_authenticated": True}


# ---------------------------------------------------------------- what they may do

_messages_today: dict[tuple[str, str], int] = {}


def count_message(identity: str, today: str | None = None) -> None:
    """Allow DAILY_MESSAGES per friend per day. Counted in memory, so a restart resets it."""
    if identity == OWNER:
        return
    today = today or date.today().isoformat()
    used = _messages_today.get((identity, today), 0)
    if used >= DAILY_MESSAGES:
        raise Auth.exceptions.HTTPException(
            status_code=429,
            detail=f"You have used today's {DAILY_MESSAGES} messages. SympAI runs on a free allowance shared with friends. "
            "Please come back tomorrow. If this cannot wait, call your doctor's office, and for an emergency call 911.",
        )
    _messages_today[(identity, today)] = used + 1


@auth.on.threads
async def own_threads_only(ctx: Auth.types.AuthContext, value: dict) -> dict:
    owner = ctx.user.identity
    if ctx.action == "create_run":
        count_message(owner)
    if isinstance(value, dict):
        # New conversations and runs are stamped with the owner.
        value.setdefault("metadata", {})
        if isinstance(value["metadata"], dict):
            value["metadata"]["owner"] = owner
    # And every read, search, update, delete or run only sees threads with that stamp.
    return {"owner": owner}


@auth.on.assistants.read
async def read_the_assistant(ctx: Auth.types.AuthContext, value: dict) -> bool:
    return True


@auth.on.assistants.search
async def find_the_assistant(ctx: Auth.types.AuthContext, value: dict) -> bool:
    return True


@auth.on
async def deny_everything_else(ctx: Auth.types.AuthContext, value: dict) -> bool:
    """Changing the assistant, crons, the server's store, and anything added later."""
    raise Auth.exceptions.HTTPException(status_code=403, detail="Not allowed.")

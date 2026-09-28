"""
Step 9. Voice. Talk to SympAI instead of typing, in the language you speak.

    listen       records from the Mac's microphone until you press Enter again
    transcribe   sends the recording to Whisper on Groq, which returns the words
                 and the language it heard, so it can reply in that language
    speak        reads the reply aloud with the Mac's built-in voices (the say command)

This matters for the people SympAI is for, including older relatives who
would rather talk than type, in Telugu, Hindi or anything else. Whisper works
out the language on its own, so nobody has to pick one from a menu.

Privacy. The recording goes to Groq, with Zero Data Retention on, and is not
kept on the laptop. Speaking the reply happens entirely on the Mac.
"""

import io
import os
import re
import subprocess
import sys
import threading
import wave

WHISPER_MODEL = "whisper-large-v3-turbo"  # 8 hours of audio a day on Groq's free tier
SAMPLE_RATE = 16000  # what Whisper works in, and small enough to upload fast
MAX_SECONDS = 120

# Whisper names the language in English. macOS voices are tagged with locale codes.
LOCALES = {
    "english": "en", "hindi": "hi", "telugu": "te", "tamil": "ta", "kannada": "kn", "malayalam": "ml",
    "marathi": "mr", "bengali": "bn", "gujarati": "gu", "urdu": "ur", "spanish": "es", "french": "fr",
    "german": "de", "chinese": "zh", "japanese": "ja", "korean": "ko", "arabic": "ar", "portuguese": "pt",
}


def to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono samples in a WAV header."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm)
    return buffer.getvalue()


def listen() -> bytes:
    """Record until Enter is pressed. Returns a WAV file as bytes.

    sounddevice talks to the microphone through PortAudio, which its macOS
    package already includes. The first time, macOS asks to allow Terminal
    to use the microphone.
    """
    import sounddevice  # imported here so the rest of SympAI works without it

    chunks: list[bytes] = []
    stop = threading.Event()

    def collect(indata, frames, time, status):
        if not stop.is_set():
            chunks.append(bytes(indata))

    with sounddevice.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=collect):
        print("   . listening, press Enter when you are done", flush=True)
        timer = threading.Timer(MAX_SECONDS, stop.set)
        timer.daemon = True  # never keeps the program alive after you quit
        timer.start()
        sys.stdin.readline()
        stop.set()
    return to_wav(b"".join(chunks))


def transcribe(audio: bytes, client=None, filename: str = "speech.wav") -> tuple[str, str]:
    """Whisper on Groq. Returns the words and the language it heard, such as ("...", "telugu").

    The terminal records WAV. A browser records WebM or MP4, and the file name's
    extension is how Whisper knows which one it got.
    """
    if client is None:
        from groq import Groq

        client = Groq(max_retries=2)
    result = client.audio.transcriptions.create(
        file=(filename, audio), model=WHISPER_MODEL, response_format="verbose_json", temperature=0
    )
    language = (getattr(result, "language", None) or "english").lower()
    return result.text.strip(), language


def for_speech(text: str) -> str:
    """Turn a written answer into something pleasant to hear."""
    text = re.sub(r"https?://\S+", "the link on screen", text)
    text = re.sub(r"[*#`>|]", "", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


# Apple ships joke voices next to the real ones, all tagged en_US. The first
# English voice alphabetically is "Albert", a croaky novelty voice, and the
# first version of this file picked it. For a health app that reads out
# serious words, that was frightening, so these can never be chosen.
NOVELTY_VOICES = {
    "albert", "bad news", "bahh", "bells", "boing", "bubbles", "cellos", "deranged", "good news",
    "hysterical", "jester", "organ", "pipe organ", "superstar", "trinoids", "whisper", "wobble",
    "zarvox", "fred", "junior", "ralph", "kathy", "princess", "agnes", "bruce", "vicki", "victoria",
}
# Newer Macs also have these for many languages. They are fine, but flat, so they come last.
FLAT_VOICES = {"eddy", "flo", "grandma", "grandpa", "reed", "rocko", "sandy", "shelley"}


def _base_name(voice: str) -> str:
    """'Eddy (English (US))' and 'Lekha (Enhanced)' become 'eddy' and 'lekha'."""
    return voice.split(" (")[0].strip().lower()


def pick_voice(language: str, voices_listing: str) -> str | None:
    """Choose a calm, natural voice for the language, from the output of `say -v ?`.

    English returns None on purpose. None means "use the Mac's system voice",
    the one you chose in System Settings, which is always a natural voice.
    Set SYMPAI_VOICE=Samantha (or any name from `say -v ?`) to pick your own.
    """
    locale = LOCALES.get(language.lower())
    if not locale or locale == "en":
        return None
    candidates = []
    for line in voices_listing.splitlines():
        match = re.match(r"^(.+?)\s+([a-z]{2})_[A-Z]{2}\s+#", line)
        if not match or match.group(2) != locale:
            continue
        name = match.group(1).strip()
        if _base_name(name) in NOVELTY_VOICES:
            continue
        # Lower rank is better. Enhanced and Premium voices sound the most human.
        rank = 0 if ("Enhanced" in name or "Premium" in name) else 2 if _base_name(name) in FLAT_VOICES else 1
        candidates.append((rank, name))
    return min(candidates)[1] if candidates else None


def speak(text: str, language: str = "english") -> bool:
    """Read the reply aloud on a Mac. Returns False where that is not possible."""
    if sys.platform != "darwin":
        return False
    locale = LOCALES.get(language.lower(), "")
    if locale in ("en", "") and os.getenv("SYMPAI_VOICE"):
        voice = os.environ["SYMPAI_VOICE"]
    else:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        voice = pick_voice(language, listing)
        if voice is None and locale not in ("en", ""):
            # An English voice reading Telugu letters sounds broken, so say nothing instead.
            print(f"   . no {language} voice is installed. System Settings, Accessibility, Spoken Content adds one.")
            return False
    command = ["say", "-v", voice] if voice else ["say"]
    subprocess.run([*command, for_speech(text)])
    return True

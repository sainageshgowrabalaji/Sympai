"""Voice, tested without a microphone or a network. The recording and Whisper are faked."""

import wave
from io import BytesIO
from types import SimpleNamespace

from sympai.graph import system_prompt
from sympai.voice import for_speech, pick_voice, to_wav, transcribe

# A few lines in the shape `say -v ?` prints on a Mac.
# The joke voices come first on purpose, because on a real Mac they do.
VOICES = """Albert              en_US    # Hello! My name is Albert.
Alex                en_US    # Most people recognize me by my voice.
Bad News            en_US    # The light you see at the end of the tunnel is the headlamp of a fast approaching train.
Eddy (Hindi (India)) hi_IN   # नमस्ते, मेरा नाम एडी है।
Lekha               hi_IN    # नमस्ते, मेरा नाम लेखा है।
Geeta               te_IN    # హలో, నా పేరు గీత.
Monica              es_MX    # Hola, me llamo Mónica.
Paulina (Enhanced)  es_MX    # Hola, me llamo Paulina.
Zarvox              en_US    # That looks like a peaceful planet."""


def test_the_recording_is_a_valid_wav():
    wav = to_wav(b"\x00\x00" * 16000)  # one second of silence
    with wave.open(BytesIO(wav)) as audio:
        assert (audio.getnchannels(), audio.getframerate(), audio.getnframes()) == (1, 16000, 16000)


def test_whisper_gives_the_words_and_the_language():
    class FakeGroq:
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kwargs):
                    assert kwargs["model"] == "whisper-large-v3-turbo"
                    return SimpleNamespace(text=" నాకు తలనొప్పి ఉంది ", language="Telugu")

    assert transcribe(b"wav", client=FakeGroq) == ("నాకు తలనొప్పి ఉంది", "telugu")


def test_a_reply_is_cleaned_up_before_it_is_read_aloud():
    written = "**Most likely** a cold.\n- Rest\n- Fluids\nSee https://medlineplus.gov/commoncold.html"
    assert for_speech(written) == "Most likely a cold. Rest Fluids See the link on screen"


def test_a_voice_is_chosen_for_the_language_heard():
    assert pick_voice("telugu", VOICES) == "Geeta"
    assert pick_voice("hindi", VOICES) == "Lekha"  # a natural voice before the flat Eddy
    assert pick_voice("spanish", VOICES) == "Paulina (Enhanced)"  # the most human one wins
    assert pick_voice("swahili", VOICES) is None


def test_english_uses_the_system_voice_never_a_joke_voice():
    # None means `say` runs with no -v, so the Mac's own system voice speaks.
    assert pick_voice("english", VOICES) is None
    only_jokes = "\n".join(line for line in VOICES.splitlines() if line.startswith(("Albert", "Bad News", "Zarvox")))
    assert pick_voice("english", only_jokes) is None


def test_the_coordinator_is_told_to_reply_in_the_language_heard():
    assert "Reply in telugu" in system_prompt({"messages": [], "language": "telugu"})
    assert "Reply in" not in system_prompt({"messages": [], "language": "english"})

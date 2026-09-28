// Reads SympAI's replies aloud with the browser's own voices. Nothing leaves the device.
//
// On a Mac the browser lists the same voices as the say command, including
// Apple's joke voices. The terminal version once picked "Albert" and it was
// frightening, so the same block list applies here.

const NOVELTY = new Set([
  "albert", "bad news", "bahh", "bells", "boing", "bubbles", "cellos", "deranged", "good news", "hysterical",
  "jester", "organ", "pipe organ", "superstar", "trinoids", "whisper", "wobble", "zarvox", "fred", "junior",
  "ralph", "kathy", "princess", "agnes", "bruce", "vicki", "victoria",
]);
const FLAT = new Set(["eddy", "flo", "grandma", "grandpa", "reed", "rocko", "sandy", "shelley"]);
const PREFERRED = ["samantha", "google us english", "microsoft aria", "microsoft jenny", "ava", "allison", "karen", "daniel"];

// Whisper names the language in English. Browser voices use tags like te-IN.
const LOCALES: Record<string, string> = {
  english: "en", hindi: "hi", telugu: "te", tamil: "ta", kannada: "kn", malayalam: "ml", marathi: "mr",
  bengali: "bn", gujarati: "gu", urdu: "ur", spanish: "es", french: "fr", german: "de", chinese: "zh",
  japanese: "ja", korean: "ko", arabic: "ar", portuguese: "pt",
};

function baseName(voice: SpeechSynthesisVoice) {
  return voice.name.split(" (")[0].trim().toLowerCase();
}

export function canSpeak() {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

export function pickVoice(language: string): SpeechSynthesisVoice | null {
  if (!canSpeak()) return null;
  const locale = LOCALES[language.toLowerCase()] || "en";
  const voices = window.speechSynthesis
    .getVoices()
    .filter((v) => v.lang.toLowerCase().startsWith(locale) && !NOVELTY.has(baseName(v)));
  if (!voices.length) return null;
  const score = (v: SpeechSynthesisVoice) => {
    const name = v.name.toLowerCase();
    if (/enhanced|premium|natural|neural/.test(name)) return 0;
    const preferred = PREFERRED.findIndex((p) => name.startsWith(p));
    if (preferred >= 0) return 1 + preferred / 100;
    if (v.default) return 2;
    if (FLAT.has(baseName(v))) return 4;
    return 3;
  };
  return [...voices].sort((a, b) => score(a) - score(b))[0];
}

export function forSpeech(markdown: string) {
  return markdown
    .replace(/https?:\/\/\S+/g, "the link on screen")
    .replace(/[*_#`>|]/g, "")
    .replace(/^\s*[-•]\s*/gm, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function speak(text: string, language = "english") {
  if (!canSpeak() || !text.trim()) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(forSpeech(text));
  const voice = pickVoice(language);
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
  } else if (language !== "english") {
    return; // an English voice reading Telugu letters sounds broken, so stay quiet instead
  }
  utterance.rate = 0.97;
  window.speechSynthesis.speak(utterance);
}

export function stopSpeaking() {
  if (canSpeak()) window.speechSynthesis.cancel();
}

"use client";

import { useEffect, useRef, useState } from "react";
import { Mic, Send, Stop } from "./icons";

type Props = {
  disabled: boolean;
  draft: string;
  setDraft: React.Dispatch<React.SetStateAction<string>>;
  onSend: (text: string, language?: string) => void;
  onTranscribe: (audio: Blob) => Promise<{ text: string; language: string }>;
};

const MAX_SECONDS = 120;

function recorderType() {
  if (typeof MediaRecorder === "undefined") return "";
  for (const type of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"]) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

export function Composer({ disabled, draft, setDraft, onSend, onTranscribe }: Props) {
  const [recording, setRecording] = useState(false);
  const [hearing, setHearing] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [voiceError, setVoiceError] = useState("");
  const [language, setLanguage] = useState<string | undefined>();
  const recorder = useRef<MediaRecorder | null>(null);
  const box = useRef<HTMLTextAreaElement | null>(null);
  // Checked after the page loads, because the first HTML is built ahead of time, with no browser to ask.
  const [canRecord, setCanRecord] = useState(false);
  useEffect(() => setCanRecord(!!navigator.mediaDevices?.getUserMedia && !!recorderType()), []);

  // The box grows with the text, up to a point.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [draft]);

  useEffect(() => {
    if (!recording) return;
    const timer = window.setInterval(() => {
      setSeconds((s) => {
        if (s + 1 >= MAX_SECONDS) recorder.current?.stop();
        return s + 1;
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [recording]);

  async function startRecording() {
    setVoiceError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const type = recorderType();
      const rec = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
      const chunks: BlobPart[] = [];
      rec.ondataavailable = (event) => event.data.size && chunks.push(event.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        setRecording(false);
        const audio = new Blob(chunks, { type: rec.mimeType || type || "audio/webm" });
        if (audio.size < 1500) return; // a tap, not speech
        setHearing(true);
        try {
          const heard = await onTranscribe(audio);
          if (heard.text) {
            setDraft((current) => (current ? `${current} ${heard.text}` : heard.text));
            setLanguage(heard.language);
            box.current?.focus();
          } else {
            setVoiceError("I did not catch any words. Please try again.");
          }
        } catch (error) {
          setVoiceError(error instanceof Error ? error.message : "Could not hear that. Please type it instead.");
        } finally {
          setHearing(false);
        }
      };
      recorder.current = rec;
      setSeconds(0);
      rec.start();
      setRecording(true);
    } catch {
      setVoiceError("The microphone is blocked. Allow it in your browser settings, or type instead.");
    }
  }

  function send() {
    const text = draft.trim();
    if (!text || disabled) return;
    onSend(text, language);
    setDraft("");
    setLanguage(undefined);
  }

  return (
    <div className="composer-wrap">
      {(recording || hearing || voiceError) && (
        <p className={`voice-note ${voiceError ? "error" : ""}`} role="status">
          {recording && <><span className="rec-dot" /> Listening, {seconds}s. Tap stop when you are done.</>}
          {hearing && "Writing down what you said..."}
          {voiceError}
        </p>
      )}
      <div className="composer">
        <textarea
          ref={box}
          rows={1}
          value={draft}
          placeholder={recording ? "Listening..." : "Describe what is going on"}
          aria-label="Your message"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            }
          }}
        />
        {canRecord && (
          <button
            type="button"
            className={`icon-button mic ${recording ? "on" : ""}`}
            onClick={() => (recording ? recorder.current?.stop() : startRecording())}
            disabled={hearing || (disabled && !recording)}
            aria-label={recording ? "Stop recording" : "Speak instead of typing"}
            title={recording ? "Stop" : "Speak"}
          >
            {recording ? <Stop /> : <Mic />}
          </button>
        )}
        <button type="button" className="send" onClick={send} disabled={disabled || !draft.trim()} aria-label="Send">
          <Send />
        </button>
      </div>
      <p className="fineprint">SympAI is not a doctor. In an emergency, call 911.</p>
    </div>
  );
}

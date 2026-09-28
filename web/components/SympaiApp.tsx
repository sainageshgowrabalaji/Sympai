"use client";

import type { Thread } from "@langchain/langgraph-sdk";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ASSISTANT, DEFAULT_API, KeyRejected, cleanUrl, forgetMe, friendlyError, isOnline, makeClient, transcribe, whoAmI,
  type ConsultState, type Me, type Remembered,
} from "@/lib/api";
import { canSpeak, speak, stopSpeaking } from "@/lib/speech";
import * as saved from "@/lib/storage";
import { lastAnswer, titleFor, toItems } from "@/lib/view";
import { Composer } from "./Composer";
import { Conversation } from "./Conversation";
import { EmergencyBanner } from "./Help";
import { Mark, Memory, Menu, SpeakerOff, SpeakerOn } from "./icons";
import { MemoryPanel } from "./MemoryPanel";
import { Offline } from "./Offline";
import { Sidebar } from "./Sidebar";
import { Welcome } from "./Welcome";

const OWNER = process.env.NEXT_PUBLIC_OWNER_NAME || "Sai";
type Phase = "loading" | "welcome" | "offline" | "ready";
type Result = "ready" | "offline" | "rejected" | "error";

function displayName(identity?: string) {
  if (!identity) return "there";
  if (identity === "me") return OWNER;
  return identity.charAt(0).toUpperCase() + identity.slice(1);
}

export default function SympaiApp() {
  const [phase, setPhase] = useState<Phase>("loading");
  const [apiUrl, setApiUrl] = useState(DEFAULT_API);
  const [key, setKey] = useState("");
  const [acknowledged, setAcknowledged] = useState(false);
  const [welcomeError, setWelcomeError] = useState<string>();
  const [checking, setChecking] = useState(false);
  const [serverUp, setServerUp] = useState(true);
  const [me, setMe] = useState<Me | null>(null);

  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [values, setValues] = useState<ConsultState>({});
  const [pending, setPending] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const [speakOn, setSpeakOn] = useState(false);
  const [voiceOut, setVoiceOut] = useState(false);
  const [language, setLanguage] = useState("english");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const [memory, setMemory] = useState<Remembered[] | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const client = useMemo(() => (key ? makeClient(cleanUrl(apiUrl), key) : null), [apiUrl, key]);
  const items = useMemo(() => toItems(values.messages), [values.messages]);
  const name = displayName(me?.name);

  // ------------------------------------------------------------ signing in

  const connect = useCallback(async (k: string, url: string): Promise<Result> => {
    setChecking(true);
    setWelcomeError(undefined);
    const up = await isOnline(url);
    setServerUp(up);
    if (!up) {
      setChecking(false);
      return "offline";
    }
    try {
      setMe(await whoAmI(url, k));
      return "ready";
    } catch (e) {
      if (e instanceof KeyRejected) {
        setWelcomeError(`That key did not work. Check it with ${OWNER}.`);
        return "rejected";
      }
      setWelcomeError(e instanceof Error ? e.message : "Could not reach SympAI.");
      return "error";
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    const savedKey = saved.load("key");
    const savedUrl = saved.load("api", DEFAULT_API);
    setAcknowledged(saved.load("ack") === "yes");
    setSpeakOn(saved.load("speak") === "on");
    setVoiceOut(canSpeak());
    setApiUrl(savedUrl);
    setKey(savedKey);
    if (!savedKey) {
      isOnline(savedUrl).then(setServerUp);
      setPhase("welcome");
      return;
    }
    connect(savedKey, savedUrl).then((result) =>
      setPhase(result === "ready" ? "ready" : result === "offline" ? "offline" : "welcome"),
    );
  }, [connect]);

  async function start(k: string, url: string) {
    saved.save("ack", "yes");
    setAcknowledged(true);
    const result = await connect(k, cleanUrl(url));
    if (result === "rejected" || result === "error") return;
    saved.save("key", k);
    saved.save("api", cleanUrl(url));
    setKey(k);
    setApiUrl(cleanUrl(url));
    setPhase(result === "ready" ? "ready" : "offline");
  }

  function signOut() {
    stopSpeaking();
    saved.remove("key");
    setKey("");
    setMe(null);
    setThreads([]);
    setThreadId(null);
    setValues({});
    setSidebarOpen(false);
    setPhase("welcome");
  }

  // ------------------------------------------------------------ conversations

  const refreshThreads = useCallback(async () => {
    if (!client) return;
    try {
      setThreads(await client.threads.search({ limit: 50, sortBy: "updated_at", sortOrder: "desc" }));
    } catch {
      /* the list stays as it was */
    }
  }, [client]);

  useEffect(() => {
    if (phase === "ready") refreshThreads();
  }, [phase, refreshThreads]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [items.length, pending, running]);

  function newConversation() {
    if (running) return;
    stopSpeaking();
    setThreadId(null);
    setValues({});
    setError(null);
    setSidebarOpen(false);
  }

  async function openConversation(id: string) {
    if (!client || running) return;
    stopSpeaking();
    setSidebarOpen(false);
    setError(null);
    setThreadId(id);
    setValues({});
    try {
      const state = await client.threads.getState(id);
      setValues((state.values as ConsultState) || {});
    } catch (e) {
      setError(friendlyError(e).message);
    }
  }

  async function deleteConversation(id: string) {
    if (!client || running) return;
    if (!window.confirm("Delete this conversation? This cannot be undone.")) return;
    try {
      await client.threads.delete(id);
      if (id === threadId) newConversation();
      await refreshThreads();
    } catch (e) {
      setError(friendlyError(e).message);
    }
  }

  async function send(text: string, heardLanguage?: string) {
    if (!client || running) return;
    stopSpeaking();
    setError(null);
    setRunning(true);
    setPending(text);
    const replyLanguage = heardLanguage || language;
    if (heardLanguage) setLanguage(heardLanguage);
    let id = threadId;
    try {
      if (!id) {
        const thread = await client.threads.create({ metadata: { title: titleFor(text) } });
        id = thread.thread_id;
        setThreadId(id);
      }
      const input: Record<string, unknown> = { messages: [{ type: "human", content: text }] };
      if (heardLanguage) input.language = heardLanguage;
      let latest: ConsultState | null = null;
      // "values" sends the whole state after each step, so each status line
      // appears the moment its step finishes, and the answer arrives at the end.
      for await (const chunk of client.runs.stream(id, ASSISTANT, { input, streamMode: "values" })) {
        if (chunk.event === "values") {
          latest = chunk.data as ConsultState;
          setValues(latest);
          setPending(null);
        } else if (chunk.event === "error") {
          const data = chunk.data as { message?: string } | undefined;
          setError(friendlyError(new Error(data?.message || "error")).message);
        }
      }
      if (latest && speakOn) speak(lastAnswer(latest.messages), replyLanguage);
    } catch (e) {
      const { status, message } = friendlyError(e);
      setDraft((current) => current || text); // the person does not lose what they wrote
      if (status === 401 || status === 403) {
        setWelcomeError(message);
        setPhase("welcome");
      } else if (!(await isOnline(apiUrl))) {
        setPhase("offline");
      } else {
        setError(message);
      }
    } finally {
      setPending(null);
      setRunning(false);
      refreshThreads();
    }
  }

  // ------------------------------------------------------------ memory

  async function openMemory() {
    setMemoryOpen(true);
    setMemory(null);
    try {
      setMemory((await whoAmI(apiUrl, key)).remembers);
    } catch (e) {
      setMemory([]);
      setError(friendlyError(e).message);
    }
  }

  async function forget() {
    try {
      await forgetMe(apiUrl, key);
      setMemory([]);
    } catch (e) {
      setError(friendlyError(e).message);
    }
  }

  // ------------------------------------------------------------ screens

  if (phase === "loading") {
    return <main className="loading" aria-label="Loading"><Mark size={44} /></main>;
  }
  if (phase === "welcome") {
    return (
      <Welcome owner={OWNER} apiUrl={apiUrl} initialKey={key} acknowledged={acknowledged} error={welcomeError}
        busy={checking} offline={!serverUp} onStart={start} />
    );
  }
  if (phase === "offline") {
    return (
      <Offline
        owner={OWNER}
        onRetry={async () => {
          const result = await connect(key, apiUrl);
          if (result !== "offline") setPhase(result === "ready" ? "ready" : "welcome");
        }}
        onChangeKey={() => setPhase("welcome")}
      />
    );
  }

  const title = threadId ? String((threads.find((t) => t.thread_id === threadId)?.metadata as Record<string, unknown> | null)?.title || "Conversation") : "New conversation";

  return (
    <div className="app">
      <Sidebar open={sidebarOpen} threads={threads} current={threadId} name={name} onNew={newConversation}
        onOpen={openConversation} onDelete={deleteConversation} onSignOut={signOut} onClose={() => setSidebarOpen(false)} />
      <div className="main">
        <header className="topbar">
          <button type="button" className="icon-button only-mobile" onClick={() => setSidebarOpen(true)} aria-label="Your conversations">
            <Menu />
          </button>
          <p className="topbar-title">{title}</p>
          <div className="topbar-actions">
            {voiceOut && (
              <button
                type="button"
                className={`icon-button ${speakOn ? "on" : ""}`}
                aria-pressed={speakOn}
                aria-label={speakOn ? "Stop reading replies aloud" : "Read replies aloud"}
                title={speakOn ? "Replies are read aloud" : "Read replies aloud"}
                onClick={() => {
                  const next = !speakOn;
                  setSpeakOn(next);
                  saved.save("speak", next ? "on" : "off");
                  if (!next) stopSpeaking();
                }}
              >
                {speakOn ? <SpeakerOn /> : <SpeakerOff />}
              </button>
            )}
            <button type="button" className="icon-button labelled" onClick={openMemory}>
              <Memory /> <span>What SympAI remembers</span>
            </button>
          </div>
        </header>
        <EmergencyBanner level={values.peak} />
        <div className="scroll">
          <div className="column">
            <Conversation items={items} running={running} pending={pending} name={name} onSuggest={setDraft} />
            {error && <p className="chat-error" role="alert">{error}</p>}
            <div ref={endRef} />
          </div>
        </div>
        <div className="column composer-column">
          <Composer disabled={running} draft={draft} setDraft={setDraft} onSend={send}
            onTranscribe={(audio) => transcribe(apiUrl, key, audio)} />
        </div>
      </div>
      <MemoryPanel open={memoryOpen} lines={memory} onClose={() => setMemoryOpen(false)} onForget={forget} />
    </div>
  );
}

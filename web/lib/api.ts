// How the page talks to the SympAI server on the laptop.
//
// Two kinds of calls. The LangGraph SDK covers conversations and runs, the
// same API the Vercel chat page used. Plain fetch covers SympAI's own three
// routes in web_api.py, for memory and voice. Both send the friend's key in
// the x-api-key header, and the server checks it on every request.

import { Client } from "@langchain/langgraph-sdk";

export const DEFAULT_API = process.env.NEXT_PUBLIC_SYMPAI_API_URL || "http://localhost:2024";
export const ASSISTANT = "sympai";

export type ToolCall = { name: string; args?: Record<string, unknown>; id?: string };

export type RawMessage = {
  type: string;
  id?: string;
  content: unknown;
  name?: string;
  tool_calls?: ToolCall[];
  response_metadata?: Record<string, unknown>;
  artifact?: unknown;
};

export type ConsultState = {
  messages?: RawMessage[];
  triage?: { level?: string; past?: boolean };
  peak?: string;
};

export type Remembered = { label: string; values: string[] };
export type Me = { name: string; remembers: Remembered[] };

export class KeyRejected extends Error {}

export function makeClient(apiUrl: string, apiKey: string) {
  // One retry at most. The SDK would otherwise retry a "daily limit reached"
  // answer several times, and the person would wait for nothing.
  return new Client({ apiUrl, apiKey, callerOptions: { maxRetries: 1 } });
}

export function cleanUrl(url: string) {
  return url.trim().replace(/\/+$/, "");
}

export async function isOnline(apiUrl: string): Promise<boolean> {
  try {
    const response = await fetch(`${cleanUrl(apiUrl)}/ok`, { signal: AbortSignal.timeout(6000) });
    return response.ok;
  } catch {
    return false;
  }
}

async function call(apiUrl: string, key: string, path: string, init: RequestInit = {}) {
  const response = await fetch(`${cleanUrl(apiUrl)}${path}`, {
    ...init,
    headers: { ...(init.headers || {}), "x-api-key": key },
  });
  if (response.status === 401 || response.status === 403) throw new KeyRejected("That key did not work.");
  // The server is up but has no /sympai routes. That happens when langgraph.json
  // changed while it was running, because the server reads that file only at start.
  if (response.status === 404) {
    throw new Error("The SympAI server needs a restart to turn on this page. If you run it, stop it with Ctrl+C and start it again.");
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `SympAI answered ${response.status}.`);
  return body;
}

export function whoAmI(apiUrl: string, key: string): Promise<Me> {
  return call(apiUrl, key, "/sympai/me");
}

export function forgetMe(apiUrl: string, key: string): Promise<{ forgotten: boolean }> {
  return call(apiUrl, key, "/sympai/me/chart", { method: "DELETE" });
}

export function transcribe(apiUrl: string, key: string, audio: Blob): Promise<{ text: string; language: string }> {
  return call(apiUrl, key, "/sympai/transcribe", {
    method: "POST",
    body: audio,
    headers: { "content-type": audio.type || "audio/webm" },
  });
}

// The server's errors arrive as "HTTP 429: {"detail": "..."}". This pulls out
// the sentence meant for the person.
export function friendlyError(error: unknown): { status?: number; message: string } {
  const status = typeof error === "object" && error && "status" in error ? Number((error as { status: unknown }).status) : undefined;
  const raw = error instanceof Error ? error.message : String(error);
  const text = typeof error === "object" && error && "text" in error ? String((error as { text: unknown }).text) : raw;
  try {
    const detail = JSON.parse(text).detail;
    if (typeof detail === "string") return { status, message: detail };
  } catch {
    /* not JSON, fall through */
  }
  if (status === 401 || status === 403) return { status, message: "That key did not work." };
  if (/connect|fetch|network/i.test(raw)) return { status, message: "Could not reach SympAI." };
  return { status, message: "Something went wrong on SympAI's side. Please try again." };
}

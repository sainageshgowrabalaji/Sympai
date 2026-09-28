// Turns the conversation, as the server stores it, into what the page shows.
//
// The state has four kinds of messages. The person's own, SympAI's answers,
// SympAI's tool calls, and the tool results. Tool results are hidden, except
// when they carry data for cards, like the clinicians the NPI Registry returned.

import type { RawMessage, ToolCall } from "./api";

export type Clinician = { name: string; specialty: string; address: string; phone: string; map: string };
export type Link = { label: string; url: string };

export type Item =
  | { kind: "person"; id: string; text: string }
  | { kind: "sympai"; id: string; text: string; alert?: string }
  | { kind: "status"; id: string; text: string }
  | { kind: "clinicians"; id: string; items: Clinician[] }
  | { kind: "links"; id: string; items: Link[] };

export function asText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part === "string" ? part : part && typeof part === "object" && "text" in part ? String(part.text) : ""))
      .join("");
  }
  return "";
}

const DEPARTMENTS: Record<string, string> = { ent: "ENT", heart_lungs: "heart and lungs", bones_muscles: "bones and muscles" };

// Older conversations, from before the server wrote these lines itself.
function describe(calls: ToolCall[]): string {
  const parts = calls.map((call) => {
    const args = call.args || {};
    if (call.name === "consult_department") {
      const key = String(args.department || "").toLowerCase();
      return `asking the ${DEPARTMENTS[key] || key.replace(/_/g, " ")} department`;
    }
    if (call.name === "search_medlineplus") return "looking this up on MedlinePlus";
    if (call.name === "search_pubmed") return "searching PubMed for recent reviews";
    if (call.name === "search_clinical_trials") return "searching ClinicalTrials.gov";
    if (call.name === "find_clinicians") return `looking up clinicians near ${args.zip_code || "you"}`;
    if (call.name === "care_options") return `looking up care options near ${args.zip_code || "you"}`;
    return "checking something";
  });
  const joined = parts.length > 1 ? `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}` : parts[0] || "";
  return joined ? joined[0].toUpperCase() + joined.slice(1) : "";
}

function stripItalics(text: string) {
  return text.replace(/^_+|_+$/g, "").replace(/\.\.\.$/, "");
}

export function toItems(messages: RawMessage[] = []): Item[] {
  const items: Item[] = [];
  // Cards come back from a tool before SympAI writes its answer, but the answer
  // says "I found a few below", so they are held and shown right after it.
  let held: Item[] = [];
  const release = () => {
    items.push(...held);
    held = [];
  };
  messages.forEach((message, index) => {
    const id = message.id || `m${index}`;
    if (message.type === "human") {
      release();
      items.push({ kind: "person", id, text: asText(message.content) });
    } else if (message.type === "ai") {
      const calls = message.tool_calls || [];
      const text = asText(message.content);
      if (calls.length) {
        items.push({ kind: "status", id, text: stripItalics(text) || describe(calls) });
      } else if (text.trim()) {
        const alert = message.response_metadata?.sympai_triage;
        items.push({ kind: "sympai", id, text, alert: typeof alert === "string" ? alert : undefined });
        release();
      }
    } else if (message.type === "tool" && message.artifact && typeof message.artifact === "object") {
      const artifact = message.artifact as { kind?: string; items?: unknown[] };
      if (artifact.kind === "clinicians" && Array.isArray(artifact.items)) {
        held.push({ kind: "clinicians", id, items: artifact.items as Clinician[] });
      } else if (artifact.kind === "links" && Array.isArray(artifact.items)) {
        held.push({ kind: "links", id, items: artifact.items as Link[] });
      }
    }
  });
  release();
  return items;
}

export function lastAnswer(messages: RawMessage[] = []): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.type === "human") return "";
    if (message.type === "ai" && !(message.tool_calls || []).length) return asText(message.content);
  }
  return "";
}

export function titleFor(text: string) {
  const oneLine = text.replace(/\s+/g, " ").trim();
  return oneLine.length > 60 ? `${oneLine.slice(0, 57)}...` : oneLine;
}

export function telLink(phone: string) {
  const digits = phone.replace(/\D/g, "");
  return `tel:${digits.length === 10 ? `+1${digits}` : digits}`;
}

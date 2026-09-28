"use client";

import type { Thread } from "@langchain/langgraph-sdk";
import { Close, Mark, Message, Plus, Trash } from "./icons";

type Props = {
  open: boolean;
  threads: Thread[];
  current: string | null;
  name: string;
  onNew: () => void;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
  onSignOut: () => void;
  onClose: () => void;
};

function when(iso?: string) {
  if (!iso) return "";
  const date = new Date(iso);
  const days = Math.floor((Date.now() - date.getTime()) / 86400000);
  if (days <= 0) return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  if (days === 1) return "Yesterday";
  if (days < 7) return date.toLocaleDateString([], { weekday: "long" });
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

export function Sidebar({ open, threads, current, name, onNew, onOpen, onDelete, onSignOut, onClose }: Props) {
  return (
    <>
      <div className={`scrim ${open ? "show" : ""}`} onClick={onClose} aria-hidden="true" />
      <aside className={`sidebar ${open ? "open" : ""}`} aria-label="Conversations">
        <div className="sidebar-top">
          <div className="brand-row small">
            <Mark size={28} />
            <span className="wordmark">SympAI</span>
          </div>
          <button type="button" className="icon-button only-mobile" onClick={onClose} aria-label="Close the list">
            <Close />
          </button>
        </div>
        <button type="button" className="new-chat" onClick={onNew}>
          <Plus size={17} /> New conversation
        </button>
        <p className="sidebar-label">Your conversations</p>
        <nav className="thread-list">
          {!threads.length && <p className="sidebar-empty">Nothing yet. Your conversations will show up here.</p>}
          {threads.map((thread) => {
            const title = String((thread.metadata as Record<string, unknown> | null)?.title || "Conversation");
            return (
              <div key={thread.thread_id} className={`thread ${thread.thread_id === current ? "active" : ""}`}>
                <button type="button" className="thread-open" onClick={() => onOpen(thread.thread_id)}>
                  <Message size={15} />
                  <span className="thread-title">{title}</span>
                  <span className="thread-when">{when(thread.updated_at)}</span>
                </button>
                <button
                  type="button"
                  className="thread-delete"
                  onClick={() => onDelete(thread.thread_id)}
                  aria-label={`Delete ${title}`}
                  title="Delete"
                >
                  <Trash size={15} />
                </button>
              </div>
            );
          })}
        </nav>
        <div className="sidebar-foot">
          <span>Signed in as <strong>{name}</strong></span>
          <button type="button" className="text-button" onClick={onSignOut}>Sign out</button>
        </div>
      </aside>
    </>
  );
}

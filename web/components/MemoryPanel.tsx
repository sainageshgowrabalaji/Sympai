"use client";

import { useState } from "react";
import type { Remembered } from "@/lib/api";
import { Close, Memory } from "./icons";

type Props = {
  open: boolean;
  lines: Remembered[] | null;
  onClose: () => void;
  onForget: () => Promise<void>;
};

export function MemoryPanel({ open, lines, onClose, onForget }: Props) {
  const [confirming, setConfirming] = useState(false);
  const [working, setWorking] = useState(false);
  if (!open) return null;

  return (
    <div className="panel-scrim" onClick={onClose}>
      <aside className="panel" role="dialog" aria-label="What SympAI remembers" onClick={(e) => e.stopPropagation()}>
        <div className="panel-head">
          <h2><Memory /> What SympAI remembers</h2>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close"><Close /></button>
        </div>
        <p className="panel-lead">
          So you do not have to repeat yourself, SympAI keeps a short chart about you. It is stored on the
          laptop that runs SympAI, never shared, and only this summary is sent to the AI.
        </p>
        {lines === null && <p className="panel-empty">Loading...</p>}
        {lines && !lines.length && <p className="panel-empty">Nothing yet. It fills in as you talk.</p>}
        {lines && lines.length > 0 && (
          <ul className="memory-list">
            {lines.map((group) => (
              <li key={group.label}>
                <span className="memory-label">{group.label}</span>
                <ul className={`memory-values ${group.values.length === 1 ? "single" : ""}`}>
                  {group.values.map((value) => <li key={value}>{value}</li>)}
                </ul>
              </li>
            ))}
          </ul>
        )}
        {lines && lines.length > 0 && !confirming && (
          <button type="button" className="danger-outline" onClick={() => setConfirming(true)}>
            Forget everything about me
          </button>
        )}
        {confirming && (
          <div className="confirm">
            <p>This deletes your chart for good. Your conversations stay in the list until you delete them.</p>
            <div className="confirm-actions">
              <button
                type="button"
                className="danger-fill"
                disabled={working}
                onClick={async () => {
                  setWorking(true);
                  await onForget();
                  setWorking(false);
                  setConfirming(false);
                }}
              >
                {working ? "Forgetting..." : "Yes, forget it"}
              </button>
              <button type="button" className="text-button" onClick={() => setConfirming(false)}>Cancel</button>
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}

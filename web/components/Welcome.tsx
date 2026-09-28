"use client";

import { useState } from "react";
import { Mark } from "./icons";

type Props = {
  owner: string;
  apiUrl: string;
  initialKey: string;
  acknowledged: boolean;
  error?: string;
  busy: boolean;
  offline: boolean;
  onStart: (key: string, apiUrl: string) => void;
};

export function Welcome({ owner, apiUrl, initialKey, acknowledged, error, busy, offline, onStart }: Props) {
  const [key, setKey] = useState(initialKey);
  const [url, setUrl] = useState(apiUrl);
  const [understood, setUnderstood] = useState(acknowledged);
  const ready = understood && key.trim().length > 10 && url.trim().length > 0 && !busy;

  return (
    <main className="welcome">
      <div className="welcome-inner">
        <div className="brand-row">
          <Mark size={40} />
          <span className="wordmark">SympAI</span>
        </div>
        <h1 className="welcome-title">A calm guide for health questions.</h1>
        <p className="welcome-lead">
          Built as a small hospital of AI agents. A coordinator listens, eight specialist departments weigh in, and a
          research desk checks real sources.
        </p>

        <ol className="how">
          <li><span>1</span>Tell it what is going on, for you or for someone you care about.</li>
          <li><span>2</span>It asks one short question at a time, then tells you what it might be and how sure it is.</li>
          <li><span>3</span>It points you to the right care, from a pharmacy to a specialist, and finds real clinicians near you.</li>
        </ol>

        <form
          className="key-form"
          onSubmit={(event) => {
            event.preventDefault();
            if (ready) onStart(key.trim(), url.trim());
          }}
        >
          <label className="check">
            <input type="checkbox" checked={understood} onChange={(e) => setUnderstood(e.target.checked)} />
            <span>
              I understand SympAI is {owner}&apos;s personal experiment. It is not a doctor, it can be wrong, and for an
              emergency I will call 911.
            </span>
          </label>

          <label className="field">
            <span className="field-label">Your SympAI key</span>
            <input
              type="password"
              autoComplete="off"
              spellCheck={false}
              placeholder="sympai_..."
              value={key}
              onChange={(e) => setKey(e.target.value)}
            />
            <span className="field-help">{owner} sends each person their own key. It stays on this device.</span>
          </label>

          <details className="advanced">
            <summary>Server address</summary>
            <input
              type="text"
              inputMode="url"
              autoCapitalize="off"
              autoCorrect="off"
              spellCheck={false}
              placeholder="https://your-mac.tail1234.ts.net"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
            <span className="field-help">Only change this if {owner} sent you a different address.</span>
          </details>

          {error && <p className="form-error" role="alert">{error}</p>}
          {offline && !error && (
            <p className="form-note">SympAI is offline right now. You can still save your key and try again later.</p>
          )}

          <button className="primary" type="submit" disabled={!ready}>
            {busy ? "Checking your key..." : "Start"}
          </button>
        </form>
      </div>
    </main>
  );
}

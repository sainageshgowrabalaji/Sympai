"use client";

import { useState } from "react";
import { HelpNow } from "./Help";
import { Mark } from "./icons";

type Props = { owner: string; onRetry: () => Promise<void>; onChangeKey: () => void };

export function Offline({ owner, onRetry, onChangeKey }: Props) {
  const [trying, setTrying] = useState(false);
  return (
    <main className="welcome">
      <div className="welcome-inner">
        <div className="brand-row">
          <Mark size={40} />
          <span className="wordmark">SympAI</span>
        </div>
        <h1 className="welcome-title">SympAI is offline right now.</h1>
        <p className="welcome-lead">
          It runs on {owner}&apos;s laptop, which is off or asleep at the moment. Your key is saved, so just try again later.
        </p>
        <div className="offline-actions">
          <button
            type="button"
            className="primary"
            disabled={trying}
            onClick={async () => {
              setTrying(true);
              await onRetry();
              setTrying(false);
            }}
          >
            {trying ? "Checking..." : "Try again"}
          </button>
          <button type="button" className="text-button" onClick={onChangeKey}>Change key or address</button>
        </div>
        <HelpNow />
      </div>
    </main>
  );
}

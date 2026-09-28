"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Clinician, Item, Link } from "@/lib/view";
import { telLink } from "@/lib/view";
import { Check, External, MapPin, Mark, Phone } from "./icons";

const SUGGESTIONS = [
  "I have had a headache for three days that will not go away",
  "Asking for my mom. She feels dizzy when she stands up",
  "My calf cramps most days. What could it be?",
  "How do I find a doctor who takes my insurance?",
];

function Markdown({ text }: { text: string }) {
  return (
    <div className="prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

function ClinicianCards({ items }: { items: Clinician[] }) {
  return (
    <section className="cards" aria-label="Clinicians SympAI found">
      <p className="cards-head">From the official NPI Registry. Call first to check they take your insurance and new patients.</p>
      <div className="card-grid">
        {items.map((person) => (
          <article className="place" key={`${person.name}-${person.address}`}>
            <p className="place-kind">{person.specialty || "Clinician"}</p>
            <h3 className="place-name">{person.name}</h3>
            <p className="place-address">{person.address}</p>
            {person.phone && <p className="place-phone">{person.phone}</p>}
            <div className="place-actions">
              {person.phone && (
                <a className="chip-button" href={telLink(person.phone)} aria-label={`Call ${person.name}, ${person.phone}`}>
                  <Phone size={15} /> Call
                </a>
              )}
              <a className="chip-button" href={person.map} target="_blank" rel="noopener noreferrer" aria-label={`Map for ${person.name}`}>
                <MapPin size={15} /> Map
              </a>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function LinkList({ items }: { items: Link[] }) {
  return (
    <section className="links" aria-label="Helpful links">
      {items.map((link) => (
        <a key={link.url} className="link-row" href={link.url} target="_blank" rel="noopener noreferrer">
          <span>{link.label}</span>
          <External size={15} />
        </a>
      ))}
    </section>
  );
}

type Props = {
  items: Item[];
  running: boolean;
  pending?: string | null;
  name: string;
  onSuggest: (text: string) => void;
};

export function Conversation({ items, running, pending, name, onSuggest }: Props) {
  const everything: Item[] = pending ? [...items, { kind: "person", id: "pending", text: pending }] : items;

  if (!everything.length) {
    return (
      <div className="empty">
        <p className="empty-hello">Hi {name}.</p>
        <p className="empty-ask">What is going on today?</p>
        <p className="empty-hint">
          Describe it in your own words, for yourself or for someone else. SympAI will ask one question at a time.
        </p>
        <div className="suggestions">
          {SUGGESTIONS.map((text) => (
            <button key={text} type="button" className="suggestion" onClick={() => onSuggest(text)}>
              {text}
            </button>
          ))}
        </div>
      </div>
    );
  }

  const last = everything[everything.length - 1];
  return (
    <div className="messages" aria-live="polite">
      {everything.map((item) => {
        if (item.kind === "person") {
          return (
            <div key={item.id} className="row person">
              <div className="bubble person-bubble">{item.text}</div>
            </div>
          );
        }
        if (item.kind === "status") {
          const working = running && item === last;
          return (
            <div key={item.id} className={`status ${working ? "working" : "done"}`}>
              {working ? <span className="dots" aria-hidden="true"><i /><i /><i /></span> : <Check size={14} />}
              <span>{item.text}</span>
            </div>
          );
        }
        if (item.kind === "clinicians") return <ClinicianCards key={item.id} items={item.items} />;
        if (item.kind === "links") return <LinkList key={item.id} items={item.items} />;
        const alertTone = item.alert === "988" ? "crisis" : item.alert && item.alert !== "urgent" ? "danger" : "";
        return (
          <div key={item.id} className="row sympai">
            <div className="avatar"><Mark size={26} /></div>
            <div className={`bubble sympai-bubble ${alertTone ? `alert ${alertTone}` : ""}`}>
              <Markdown text={item.text} />
            </div>
          </div>
        );
      })}
      {running && last.kind === "person" && (
        <div className="status working">
          <span className="dots" aria-hidden="true"><i /><i /><i /></span>
          <span>SympAI is thinking</span>
        </div>
      )}
    </div>
  );
}

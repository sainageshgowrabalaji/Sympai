// Emergency help, shown even when SympAI itself is offline. It needs nothing from the server.

import { Alert, Heart, MapPin, Phone } from "./icons";

export function HelpNow() {
  return (
    <section className="help-card" aria-label="If you need help now">
      <h2 className="help-title">If you need help now</h2>
      <a className="help-row danger" href="tel:911">
        <Phone /> <span><strong>Call 911</strong> for chest pain, trouble breathing, stroke signs or anything that feels like an emergency</span>
      </a>
      <a className="help-row crisis" href="tel:988">
        <Heart /> <span><strong>Call or text 988</strong> if you are thinking about hurting yourself</span>
      </a>
      <a className="help-row" href="tel:+18002221222">
        <Alert /> <span><strong>Poison Control 1-800-222-1222</strong> for anything swallowed or breathed in</span>
      </a>
      <a className="help-row" href="https://www.google.com/maps/search/?api=1&query=urgent+care+near+me" target="_blank" rel="noopener noreferrer">
        <MapPin /> <span><strong>Find urgent care near you</strong> for problems that cannot wait for an appointment</span>
      </a>
    </section>
  );
}

const BANNERS: Record<string, { tone: string; title: string; body: string; actions: { label: string; href: string }[] }> = {
  "911": {
    tone: "danger",
    title: "If this is happening now, call 911",
    body: "Do not wait for SympAI to finish. Emergency services can help right away.",
    actions: [{ label: "Call 911", href: "tel:911" }],
  },
  poison: {
    tone: "danger",
    title: "Call Poison Control now",
    body: "They answer 24 hours a day, and it is free. If the person is not breathing or will not wake up, call 911.",
    actions: [{ label: "Call 1-800-222-1222", href: "tel:+18002221222" }, { label: "Call 911", href: "tel:911" }],
  },
  "988": {
    tone: "crisis",
    title: "You do not have to go through this alone",
    body: "The 988 Lifeline is free, private and open all day and night. You can call or text.",
    actions: [{ label: "Call 988", href: "tel:988" }, { label: "Text 988", href: "sms:988" }],
  },
};

export function EmergencyBanner({ level }: { level?: string }) {
  const banner = level ? BANNERS[level] : undefined;
  if (!banner) return null;
  return (
    <div className={`banner ${banner.tone}`} role="alert">
      <div>
        <p className="banner-title">{banner.title}</p>
        <p className="banner-body">{banner.body}</p>
      </div>
      <div className="banner-actions">
        {banner.actions.map((action) => (
          <a key={action.href} className="banner-button" href={action.href}>
            <Phone size={16} /> {action.label}
          </a>
        ))}
      </div>
    </div>
  );
}

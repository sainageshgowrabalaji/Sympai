// Small line icons, drawn inline so the page loads nothing from outside.

type Props = { size?: number };

function Svg({ size = 18, children }: Props & { children: React.ReactNode }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {children}
    </svg>
  );
}

export const Mark = ({ size = 28 }: Props) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
    <rect x="1" y="1" width="30" height="30" rx="9" fill="var(--accent)" />
    <path d="M6 17h5l2.5-6 4 11 3-8 1.5 3H26" fill="none" stroke="var(--on-accent)" strokeWidth="2.2"
      strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

export const Plus = (p: Props) => <Svg {...p}><path d="M12 5v14M5 12h14" /></Svg>;
export const Trash = (p: Props) => (
  <Svg {...p}><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3" /></Svg>
);
export const Mic = (p: Props) => (
  <Svg {...p}><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></Svg>
);
export const Stop = (p: Props) => <Svg {...p}><rect x="7" y="7" width="10" height="10" rx="2" /></Svg>;
export const Send = (p: Props) => <Svg {...p}><path d="M5 12h13M13 6l6 6-6 6" /></Svg>;
export const SpeakerOn = (p: Props) => (
  <Svg {...p}><path d="M4 9h4l5-4v14l-5-4H4zM16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12" /></Svg>
);
export const SpeakerOff = (p: Props) => (
  <Svg {...p}><path d="M4 9h4l5-4v14l-5-4H4zM17 9l5 6M22 9l-5 6" /></Svg>
);
export const Memory = (p: Props) => (
  <Svg {...p}><path d="M6 4h9l3 3v13H6z" /><path d="M9 11h6M9 15h6M9 7h3" /></Svg>
);
export const Menu = (p: Props) => <Svg {...p}><path d="M4 7h16M4 12h16M4 17h16" /></Svg>;
export const Close = (p: Props) => <Svg {...p}><path d="M6 6l12 12M18 6 6 18" /></Svg>;
export const Phone = (p: Props) => (
  <Svg {...p}><path d="M5 4h4l2 5-2.5 1.5a11 11 0 0 0 5 5L15 13l5 2v4a2 2 0 0 1-2 2A16 16 0 0 1 3 6a2 2 0 0 1 2-2" /></Svg>
);
export const MapPin = (p: Props) => (
  <Svg {...p}><path d="M12 21s-7-6.2-7-11.5A7 7 0 0 1 19 9.5C19 14.8 12 21 12 21z" /><circle cx="12" cy="9.5" r="2.5" /></Svg>
);
export const External = (p: Props) => (
  <Svg {...p}><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" /></Svg>
);
export const Check = (p: Props) => <Svg {...p}><path d="M5 12.5l4.5 4.5L19 7.5" /></Svg>;
export const Alert = (p: Props) => (
  <Svg {...p}><path d="M12 3 2.5 20h19z" /><path d="M12 10v4M12 17.5v.01" /></Svg>
);
export const Heart = (p: Props) => (
  <Svg {...p}><path d="M12 20s-7.5-4.6-7.5-10A4.3 4.3 0 0 1 12 7.3 4.3 4.3 0 0 1 19.5 10c0 5.4-7.5 10-7.5 10z" /></Svg>
);
export const Message = (p: Props) => (
  <Svg {...p}><path d="M4 5h16v11H9l-5 4z" /></Svg>
);

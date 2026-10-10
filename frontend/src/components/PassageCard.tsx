import { useState, type ReactNode } from "react";
import type { Kind, Passage } from "../api";

export function KindBadge({ kind }: { kind: Kind }) {
  return kind === "note" ? (
    <span className="badge badge-note">Translator's note, not Charaka's words</span>
  ) : (
    <span className="badge badge-text">Charaka's text</span>
  );
}

interface Props {
  passage: Passage;
  label?: string; // "S1", "N1", ... when the passage is cited in an answer
  id?: string;
  open?: boolean;
  onToggle?: (open: boolean) => void;
  meta?: ReactNode; // e.g. search ranks
}

/**
 * One passage of the book: its citation, whose words it is, and the text
 * (expandable). Open/closed is controlled by the parent when it passes
 * `open`, otherwise kept here.
 */
export function PassageCard({ passage, label, id, open, onToggle, meta }: Props) {
  const [ownOpen, setOwnOpen] = useState(false);
  const isOpen = open ?? ownOpen;
  return (
    <article id={id} className={`passage passage-${passage.kind}`}>
      <header className="passage-head">
        {label && <span className="cite cite-static">{label}</span>}
        <span className="passage-citation">{passage.citation}</span>
        <KindBadge kind={passage.kind} />
      </header>
      {meta && <div className="passage-meta">{meta}</div>}
      <details
        open={isOpen}
        onToggle={(e) => {
          const nowOpen = e.currentTarget.open;
          setOwnOpen(nowOpen);
          onToggle?.(nowOpen);
        }}
      >
        <summary>{isOpen ? "Hide passage" : "Show passage"}</summary>
        <p className="passage-body">{passage.text}</p>
        <p className="passage-attribution">{passage.attribution}</p>
      </details>
    </article>
  );
}

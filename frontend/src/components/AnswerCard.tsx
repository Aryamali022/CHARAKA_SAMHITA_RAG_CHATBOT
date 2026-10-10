import { useId, useState } from "react";
import Markdown from "react-markdown";
import type { AskResponse } from "../api";
import { citedLabel, linkCitations } from "../citations";
import { PassageCard } from "./PassageCard";

/**
 * One answer from /api/ask. Everything the backend enforces is shown as sent:
 * the translator's-note warning, the uncited warning, removed citations and
 * the disclaimer. Clicking a citation chip opens that source below.
 */
export function AnswerCard({ response }: { response: AskResponse }) {
  const idPrefix = useId();
  const [openLabels, setOpenLabels] = useState<Set<string>>(new Set());
  const sourceId = (label: string) => `${idPrefix}-source-${label}`;

  const setOpen = (label: string, open: boolean) =>
    setOpenLabels((current) => {
      const next = new Set(current);
      if (open) next.add(label);
      else next.delete(label);
      return next;
    });

  const showSource = (label: string) => {
    setOpen(label, true);
    document.getElementById(sourceId(label))?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
  };

  const markdown = linkCitations(response.answer, response.sources.map((s) => s.label));

  return (
    <div className={`answer ${response.found ? "" : "answer-not-found"}`}>
      <div className="answer-text">
        <Markdown
          components={{
            a: ({ href, children }) => {
              const label = citedLabel(href);
              if (label) {
                return (
                  <button type="button" className="cite" onClick={() => showSource(label)}
                          aria-label={`Show source ${label}`}>
                    {label}
                  </button>
                );
              }
              return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
            },
          }}
        >
          {markdown}
        </Markdown>
      </div>

      {response.notes_warning && <p className="notice notice-note">{response.notes_warning}</p>}
      {response.uncited_warning && <p className="notice notice-warn">{response.uncited_warning}</p>}
      {response.removed_citations.length > 0 && (
        <p className="notice notice-info">
          Removed citations to passages that were not given: {response.removed_citations.join(", ")}
        </p>
      )}

      {response.sources.length > 0 && (
        <section className="sources" aria-label="Sources">
          <h3>Sources</h3>
          {response.sources.map((source) => (
            <PassageCard
              key={source.label}
              id={sourceId(source.label)}
              passage={source}
              label={source.label}
              open={openLabels.has(source.label)}
              onToggle={(open) => setOpen(source.label, open)}
            />
          ))}
        </section>
      )}

      <p className="disclaimer">{response.disclaimer}</p>
    </div>
  );
}

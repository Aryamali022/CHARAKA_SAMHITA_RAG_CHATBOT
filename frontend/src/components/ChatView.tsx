import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { api, ApiError, type AskResponse } from "../api";
import { AnswerCard } from "./AnswerCard";

export const EXAMPLE_QUESTIONS = [
  "Which natural urges should never be suppressed?",
  "How many bones are there in the human body?",
  "What is Rasayana?",
  "What are the causes of fever?",
  "How is excessive thirst treated?",
];

interface Exchange {
  id: number;
  question: string;
  response?: AskResponse;
  error?: string;
}

/** The "Ask" tab: a conversation of questions and cited answers. */
export function ChatView() {
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [draft, setDraft] = useState("");
  const [rewrite, setRewrite] = useState(false);
  const [textOnly, setTextOnly] = useState(false);
  const [busy, setBusy] = useState(false);
  const nextId = useRef(1);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [exchanges]);

  async function ask(question: string) {
    question = question.trim();
    if (!question || busy) return;
    const id = nextId.current++;
    setExchanges((list) => [...list, { id, question }]);
    setDraft("");
    setBusy(true);
    try {
      const response = await api.ask(question, { rewrite, textOnly });
      setExchanges((list) => list.map((e) => (e.id === id ? { ...e, response } : e)));
    } catch (error) {
      const message = error instanceof ApiError ? error.message : "Something went wrong.";
      setExchanges((list) => list.map((e) => (e.id === id ? { ...e, error: message } : e)));
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void ask(draft);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {   // Enter sends, Shift+Enter makes a new line
      event.preventDefault();
      void ask(draft);
    }
  }

  return (
    <div className="chat">
      <div className="messages" aria-live="polite">
        {exchanges.length === 0 && (
          <div className="empty">
            <p>Ask a question about the Charaka Samhita. Answers come only from the text, with sources.</p>
            <p className="examples-title">Try one of these:</p>
            <ul className="examples">
              {EXAMPLE_QUESTIONS.map((q) => (
                <li key={q}>
                  <button type="button" className="example" onClick={() => void ask(q)}>{q}</button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {exchanges.map((exchange) => (
          <div key={exchange.id} className="exchange">
            <p className="question"><span className="who">You</span>{exchange.question}</p>
            {exchange.response && <AnswerCard response={exchange.response} />}
            {exchange.error && <p className="notice notice-error" role="alert">{exchange.error}</p>}
            {!exchange.response && !exchange.error && (
              <p className="thinking" role="status">
                <span className="spinner" aria-hidden="true" />
                Searching the text and writing an answer. This can take up to a minute.
              </p>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form className="composer" onSubmit={onSubmit}>
        <label htmlFor="question" className="visually-hidden">Your question</label>
        <textarea
          id="question"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask about the Charaka Samhita…"
          rows={2}
          maxLength={1000}
        />
        <button type="submit" disabled={busy || !draft.trim()}>{busy ? "Answering…" : "Ask"}</button>
        <div className="options">
          <label>
            <input type="checkbox" checked={textOnly} onChange={(e) => setTextOnly(e.target.checked)} />
            Charaka's text only (no translator's notes)
          </label>
          <label title="Adds the old translation's vocabulary to the search. Helps everyday wording; slower.">
            <input type="checkbox" checked={rewrite} onChange={(e) => setRewrite(e.target.checked)} />
            Rewrite question for the old translation (slower)
          </label>
        </div>
      </form>
    </div>
  );
}

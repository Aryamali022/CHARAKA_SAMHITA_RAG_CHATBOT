import { useState, type FormEvent } from "react";
import { api, ApiError, type SearchHit } from "../api";
import { PassageCard } from "./PassageCard";

function ranks(hit: SearchHit): string {
  const parts = [];
  if (hit.dense_rank) parts.push(`meaning #${hit.dense_rank}`);
  if (hit.bm25_rank) parts.push(`keyword #${hit.bm25_rank}`);
  return `Found by ${parts.join(", ")}`;
}

/** The "Search the text" tab: passages only, no answer model, so it is quick. */
export function SearchView() {
  const [query, setQuery] = useState("");
  const [textOnly, setTextOnly] = useState(false);
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setHits((await api.search(query.trim(), textOnly)).results);
    } catch (e) {
      setHits(null);
      setError(e instanceof ApiError ? e.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="search">
      <form className="search-form" onSubmit={onSubmit} role="search">
        <label htmlFor="search-query" className="visually-hidden">Search the text</label>
        <input
          id="search-query"
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search the text, e.g. Rasayana, bones, thirst"
          maxLength={500}
        />
        <button type="submit" disabled={busy || !query.trim()}>{busy ? "Searching…" : "Search"}</button>
        <label className="search-option">
          <input type="checkbox" checked={textOnly} onChange={(e) => setTextOnly(e.target.checked)} />
          Charaka's text only
        </label>
      </form>

      {error && <p className="notice notice-error" role="alert">{error}</p>}
      {hits && hits.length === 0 && <p className="empty">No passages found.</p>}
      {hits && hits.length > 0 && (
        <section aria-label="Search results" className="results">
          {hits.map((hit) => (
            <PassageCard key={hit.chunk_id} passage={hit} meta={ranks(hit)} />
          ))}
        </section>
      )}
    </div>
  );
}

// Talks to the backend server (src/api.py). The types mirror its responses;
// see http://localhost:8000/docs for the full description of each field.

export type Kind = "text" | "note"; // "text" = Charaka, "note" = translator's note

export interface Passage {
  chunk_id: string;
  kind: Kind;
  attribution: string;
  sthana: string;
  sthana_name: string;
  lesson: number;
  lesson_title: string;
  citation: string;
  scan_pages: number[];
  printed_pages: string[];
  text: string;
}

export interface Source extends Passage {
  label: string; // as cited in the answer: "S1", "N1", ...
}

export interface SearchHit extends Passage {
  score: number;
  dense_rank: number | null;
  bm25_rank: number | null;
}

export interface AskResponse {
  question: string;
  found: boolean;
  answer: string;
  sources: Source[];
  uses_translator_notes: boolean;
  notes_warning: string | null;
  uncited_warning: string | null;
  removed_citations: string[];
  disclaimer: string;
}

export interface SearchResponse {
  query: string;
  results: SearchHit[];
}

export interface Health {
  status: "ok" | "degraded";
  search: { ready: boolean; store: string; error: string | null; chunks: number };
  answers: { ready: boolean; model: string | null; error: string | null };
}

export interface AskOptions {
  rewrite?: boolean;
  textOnly?: boolean;
}

/** An error the user should see, with the server's explanation when there is one. */
export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

// Empty: same server as the page (the Vite proxy in development, FastAPI in production).
const API_BASE = import.meta.env.VITE_API_URL ?? "";

async function request<T>(path: string, body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(API_BASE + path, body === undefined ? undefined : {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError("Cannot reach the backend server. Start it with: python -m src.api", 0);
  }
  if (!response.ok) {
    throw new ApiError(await errorMessage(response), response.status);
  }
  return (await response.json()) as T;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const { detail } = await response.json();
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail[0]?.msg) return `Invalid request: ${detail[0].msg}`;
  } catch {
    // not JSON: fall through
  }
  if (response.status === 502 || response.status === 504) {
    return "The backend server is not running. Start it with: python -m src.api";
  }
  return `The server returned an error (${response.status}).`;
}

export const api = {
  health: () => request<Health>("/api/health"),
  ask: (question: string, options: AskOptions = {}) =>
    request<AskResponse>("/api/ask", {
      question,
      rewrite: options.rewrite ?? false,
      text_only: options.textOnly ?? false,
    }),
  search: (query: string, textOnly = false, k = 8) =>
    request<SearchResponse>("/api/search", { query, k, text_only: textOnly }),
};

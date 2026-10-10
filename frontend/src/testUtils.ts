// A fake backend for the component tests: replaces fetch, records requests.
import { vi } from "vitest";
import type { AskResponse, Passage, SearchHit, Source } from "./api";

export function passage(overrides: Partial<Passage> = {}): Passage {
  return {
    chunk_id: "sutra-07-text-001", kind: "text", attribution: "Charaka Samhita, tr. A. C. Kaviratna",
    sthana: "sutra", sthana_name: "Sutra Sthana", lesson: 7, lesson_title: "Navegandharaniya",
    citation: "Sutra Sthana, Lesson VII (Navegandharaniya), pp. 73-74",
    scan_pages: [80, 81], printed_pages: ["73", "74"],
    text: "One should not suppress the urge to sneeze.", ...overrides,
  };
}

export const CHARAKA_SOURCE: Source = { ...passage(), label: "S1" };
export const NOTE_SOURCE: Source = {
  ...passage({
    chunk_id: "sutra-07-note-001", kind: "note", text: "'Kshavathu' is sneezing.—T.",
    attribution: "Translator's note by A. C. Kaviratna, not part of the Charaka Samhita",
  }),
  label: "N1",
};

export function answer(overrides: Partial<AskResponse> = {}): AskResponse {
  return {
    question: "Why not hold back a sneeze?", found: true,
    answer: "Never suppress a **sneeze** [S1].", sources: [CHARAKA_SOURCE],
    uses_translator_notes: false, notes_warning: null, uncited_warning: null,
    removed_citations: [],
    disclaimer: "Educational use only, not medical advice.", ...overrides,
  };
}

export const SEARCH_HIT: SearchHit = { ...passage(), score: 0.03, dense_rank: 1, bm25_rank: 4 };

type Reply = { status?: number; body: unknown } | Error;

/** Replace fetch: each path gets a reply; returns the mock to inspect requests. */
export function fakeBackend(replies: Record<string, Reply>) {
  const fetchMock = vi.fn(async (url: string, _init?: RequestInit) => {
    const reply = replies[url];
    if (reply === undefined) throw new Error(`unexpected request to ${url}`);
    if (reply instanceof Error) throw reply;
    return new Response(JSON.stringify(reply.body), {
      status: reply.status ?? 200, headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

export function requestBody(fetchMock: ReturnType<typeof fakeBackend>, call = 0) {
  const init = fetchMock.mock.calls[call][1] as RequestInit | undefined;
  return JSON.parse(String(init?.body));
}

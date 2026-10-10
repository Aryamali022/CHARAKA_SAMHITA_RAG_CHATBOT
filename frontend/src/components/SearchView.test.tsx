import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { fakeBackend, requestBody, SEARCH_HIT } from "../testUtils";
import { SearchView } from "./SearchView";

describe("SearchView", () => {
  it("shows passages with their citation, kind and how they were found", async () => {
    const fetchMock = fakeBackend({ "/api/search": { body: { query: "sneeze", results: [SEARCH_HIT] } } });
    render(<SearchView />);
    await userEvent.type(screen.getByLabelText("Search the text"), "sneeze{Enter}");

    expect(await screen.findByText("Sutra Sthana, Lesson VII (Navegandharaniya), pp. 73-74")).toBeInTheDocument();
    expect(screen.getByText("Charaka's text")).toBeInTheDocument();
    expect(screen.getByText("Found by meaning #1, keyword #4")).toBeInTheDocument();
    expect(requestBody(fetchMock)).toEqual({ query: "sneeze", k: 8, text_only: false });
  });

  it("says so when nothing is found", async () => {
    fakeBackend({ "/api/search": { body: { query: "zzz", results: [] } } });
    render(<SearchView />);
    await userEvent.type(screen.getByLabelText("Search the text"), "zzz{Enter}");
    expect(await screen.findByText("No passages found.")).toBeInTheDocument();
  });
});

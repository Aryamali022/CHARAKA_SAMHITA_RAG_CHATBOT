import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { answer, CHARAKA_SOURCE, NOTE_SOURCE } from "../testUtils";
import { AnswerCard } from "./AnswerCard";

describe("AnswerCard", () => {
  it("renders the answer as formatted text with a citation chip", () => {
    render(<AnswerCard response={answer()} />);
    expect(screen.getByText("sneeze", { selector: "strong" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show source S1" })).toBeInTheDocument();
  });

  it("opens the cited passage when its chip is clicked", async () => {
    render(<AnswerCard response={answer()} />);
    const card = screen.getByRole("article");
    expect(within(card).getByText(/One should not suppress/).closest("details")).not.toHaveAttribute("open");
    await userEvent.click(screen.getByRole("button", { name: "Show source S1" }));
    expect(within(card).getByText(/One should not suppress/).closest("details")).toHaveAttribute("open");
  });

  it("marks translator's notes and shows the backend's warning", () => {
    render(<AnswerCard response={answer({
      answer: "Kshavathu means sneezing [N1], as Charaka says [S1].",
      sources: [NOTE_SOURCE, CHARAKA_SOURCE], uses_translator_notes: true,
      notes_warning: "Note: parts of this answer come from the translator's notes.",
    })} />);
    expect(screen.getByText("Note: parts of this answer come from the translator's notes.")).toBeInTheDocument();
    expect(screen.getByText("Translator's note, not Charaka's words")).toBeInTheDocument();
    expect(screen.getByText("Charaka's text")).toBeInTheDocument();
  });

  it("shows the 'not covered' reply without sources, and always the disclaimer", () => {
    render(<AnswerCard response={answer({
      found: false, answer: "The Charaka Samhita does not answer this.", sources: [],
    })} />);
    expect(screen.getByText("The Charaka Samhita does not answer this.")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Sources" })).not.toBeInTheDocument();
    expect(screen.getByText("Educational use only, not medical advice.")).toBeInTheDocument();
  });

  it("reports removed citations and uncited answers", () => {
    render(<AnswerCard response={answer({
      answer: "Sneezing is natural.", sources: [], removed_citations: ["S9"],
      uncited_warning: "Warning: the model cited no passage.",
    })} />);
    expect(screen.getByText(/Removed citations .*: S9/)).toBeInTheDocument();
    expect(screen.getByText("Warning: the model cited no passage.")).toBeInTheDocument();
  });
});

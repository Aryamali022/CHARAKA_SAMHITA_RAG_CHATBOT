import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { answer, fakeBackend, requestBody } from "../testUtils";
import { ChatView, EXAMPLE_QUESTIONS } from "./ChatView";

describe("ChatView", () => {
  it("sends the question with the chosen options and shows the answer", async () => {
    const fetchMock = fakeBackend({ "/api/ask": { body: answer() } });
    render(<ChatView />);

    await userEvent.click(screen.getByLabelText(/Charaka's text only/));
    await userEvent.type(screen.getByLabelText("Your question"), "Why not hold back a sneeze?{Enter}");

    expect(await screen.findByRole("button", { name: "Show source S1" })).toBeInTheDocument();
    expect(screen.getByText("Why not hold back a sneeze?")).toBeInTheDocument();
    expect(requestBody(fetchMock)).toEqual({
      question: "Why not hold back a sneeze?", rewrite: false, text_only: true,
    });
    expect(screen.getByLabelText("Your question")).toHaveValue("");
  });

  it("asks an example question when it is clicked", async () => {
    const fetchMock = fakeBackend({ "/api/ask": { body: answer() } });
    render(<ChatView />);
    await userEvent.click(screen.getByRole("button", { name: EXAMPLE_QUESTIONS[0] }));
    await screen.findByRole("button", { name: "Show source S1" });
    expect(requestBody(fetchMock).question).toBe(EXAMPLE_QUESTIONS[0]);
  });

  it("shows the server's explanation when the answer fails", async () => {
    fakeBackend({ "/api/ask": { status: 503, body: { detail: "Rate limit reached. Wait a minute and try again." } } });
    render(<ChatView />);
    await userEvent.type(screen.getByLabelText("Your question"), "Sneeze?{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("Rate limit reached");
  });

  it("explains how to start the backend when it cannot be reached", async () => {
    fakeBackend({ "/api/ask": new TypeError("Failed to fetch") });
    render(<ChatView />);
    await userEvent.type(screen.getByLabelText("Your question"), "Sneeze?{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("python -m src.api");
  });

  it("does not send an empty question", async () => {
    const fetchMock = fakeBackend({});
    render(<ChatView />);
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Your question"), "   {Enter}");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

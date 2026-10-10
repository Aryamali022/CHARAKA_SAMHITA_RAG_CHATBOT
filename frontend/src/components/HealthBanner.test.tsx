import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { fakeBackend } from "../testUtils";
import { HealthBanner } from "./HealthBanner";

const HEALTHY = {
  status: "ok",
  search: { ready: true, store: "Qdrant server", error: null, chunks: 2296 },
  answers: { ready: true, model: "openai/gpt-oss-20b", error: null },
};

describe("HealthBanner", () => {
  it("shows nothing when everything is ready", async () => {
    const fetchMock = fakeBackend({ "/api/health": { body: HEALTHY } });
    const { container } = render(<HealthBanner />);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(fetchMock).toHaveBeenCalled();
    expect(container).toBeEmptyDOMElement();
  });

  it("says which part is not ready and why", async () => {
    fakeBackend({ "/api/health": { body: {
      ...HEALTHY, status: "degraded",
      answers: { ready: false, model: null, error: "NVIDIA_API_KEY is not set." },
    } } });
    render(<HealthBanner />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Answers are not available: NVIDIA_API_KEY is not set.");
  });

  it("explains how to start the backend when it is down", async () => {
    fakeBackend({ "/api/health": new TypeError("Failed to fetch") });
    render(<HealthBanner />);
    expect(await screen.findByRole("alert")).toHaveTextContent("python -m src.api");
  });
});

import { describe, expect, it } from "vitest";
import { citedLabel, linkCitations } from "./citations";

describe("linkCitations", () => {
  it("turns citations to returned sources into links", () => {
    expect(linkCitations("Never suppress it [S1][N2].", ["S1", "N2"]))
      .toBe("Never suppress it [S1](#cite-S1)[N2](#cite-N2).");
  });

  it("leaves citations to unknown sources and ordinary brackets alone", () => {
    expect(linkCitations("See [S9] and [as listed].", ["S1"])).toBe("See [S9] and [as listed].");
  });
});

describe("citedLabel", () => {
  it("reads the label from a citation link only", () => {
    expect(citedLabel("#cite-N1")).toBe("N1");
    expect(citedLabel("https://example.org")).toBeNull();
    expect(citedLabel(undefined)).toBeNull();
  });
});

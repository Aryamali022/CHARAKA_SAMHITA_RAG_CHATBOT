import { useState } from "react";
import { ChatView } from "./components/ChatView";
import { HealthBanner } from "./components/HealthBanner";
import { SearchView } from "./components/SearchView";

type Tab = "ask" | "search";

export default function App() {
  const [tab, setTab] = useState<Tab>("ask");

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Charaka Samhita Chatbot</h1>
          <p className="subtitle">
            Answers from the English translation by A. C. Kaviratna (Calcutta, 1890 onward), with sources
          </p>
        </div>
        <nav className="tabs" aria-label="Mode">
          <button type="button" aria-pressed={tab === "ask"} onClick={() => setTab("ask")}>Ask</button>
          <button type="button" aria-pressed={tab === "search"} onClick={() => setTab("search")}>
            Search the text
          </button>
        </nav>
      </header>

      <p className="safety">
        Educational use only, not medical advice. It does not diagnose or recommend treatment.
        Consult a qualified practitioner.
      </p>
      <HealthBanner />

      <main>
        {/* Both stay mounted, so switching tabs keeps the conversation and the results. */}
        <div hidden={tab !== "ask"}><ChatView /></div>
        <div hidden={tab !== "search"}><SearchView /></div>
      </main>

      <footer className="bottom">
        Text: Source Library transcription of the Kaviratna translation, CC BY-SA 4.0.
        Translator's notes are marked as such and are not Charaka's words.
      </footer>
    </div>
  );
}

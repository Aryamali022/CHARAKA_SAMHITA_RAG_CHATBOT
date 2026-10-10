import { useEffect, useState } from "react";
import { api, ApiError, type Health } from "../api";

/** Shown only when something is wrong: the backend is down, or a part of it is not ready. */
export function HealthBanner() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.health()
      .then(setHealth)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Cannot reach the backend server."));
  }, []);

  if (error) {
    return <div className="banner banner-error" role="alert">{error}</div>;
  }
  if (!health || health.status === "ok") return null;
  return (
    <div className="banner banner-warn" role="alert">
      {!health.search.ready && <p>Search is not available: {health.search.error}</p>}
      {!health.answers.ready && <p>Answers are not available: {health.answers.error}</p>}
    </div>
  );
}

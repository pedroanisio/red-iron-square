import type { ReactElement } from "react";

export function StatusBanner({
  line,
  socketStatus,
  audioFallback,
  errorMessage,
}: {
  line: string;
  socketStatus: string;
  audioFallback: string | null;
  errorMessage: string | null;
}): ReactElement {
  return (
    <div className="status-banner" aria-live="polite">
      <p>{line}</p>
      <p className="status-meta">
        Connection: <strong>{socketStatus}</strong>
      </p>
      {errorMessage ? <p className="status-fallback">{errorMessage}</p> : null}
      {audioFallback ? (
        <p className="status-fallback">{audioFallback}</p>
      ) : null}
    </div>
  );
}

import type { ReactElement } from "react";

export function StatusBanner({
  line,
  socketStatus,
  audioFallback,
}: {
  line: string;
  socketStatus: string;
  audioFallback: string | null;
}): ReactElement {
  return (
    <div className="status-banner">
      <p>{line}</p>
      <p className="status-meta">
        Connection: <strong>{socketStatus}</strong>
      </p>
      {audioFallback ? (
        <p className="status-fallback">{audioFallback}</p>
      ) : null}
    </div>
  );
}

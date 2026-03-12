import type { ReactElement } from "react";

/** Animated indicator shown while an agent is actively speaking. */
export function ThinkingState({
  visible,
  label,
}: {
  visible: boolean;
  label: string;
}): ReactElement | null {
  if (!visible) {
    return null;
  }
  return (
    <div className="thinking-state" aria-live="polite">
      <span>{label}</span>
      <span className="thinking-dots" aria-hidden="true" />
    </div>
  );
}

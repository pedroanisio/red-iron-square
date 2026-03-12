import type { ReactElement } from "react";

type StateBarProps = {
  label: string;
  value: number;
  tone: "mood" | "energy" | "calm";
};

/** Horizontal progress bar that visualises a single agent dimension (mood, energy, or calm). */
export function StateBar({ label, value, tone }: StateBarProps): ReactElement {
  const percent = `${Math.round(((value + (tone === "mood" ? 1 : 0)) / (tone === "mood" ? 2 : 1)) * 100)}%`;
  const min = tone === "mood" ? -1 : 0;
  const max = 1;
  const boundedValue = Math.max(min, Math.min(max, value));
  return (
    <div className="state-bar">
      <div className="state-bar-label">
        <span>{label}</span>
        <span>{percent}</span>
      </div>
      <div
        className={`state-bar-track state-bar-track-${tone}`}
        role="progressbar"
        aria-label={label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={boundedValue}
      >
        <div className="state-bar-fill" style={{ width: percent }} />
      </div>
    </div>
  );
}

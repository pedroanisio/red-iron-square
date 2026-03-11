import type { ReactElement } from "react";

type StateBarProps = {
  label: string;
  value: number;
  tone: "mood" | "energy" | "calm";
};

export function StateBar({ label, value, tone }: StateBarProps): ReactElement {
  const percent = `${Math.round(((value + (tone === "mood" ? 1 : 0)) / (tone === "mood" ? 2 : 1)) * 100)}%`;
  return (
    <div className="state-bar">
      <div className="state-bar-label">
        <span>{label}</span>
        <span>{percent}</span>
      </div>
      <div className={`state-bar-track state-bar-track-${tone}`}>
        <div className="state-bar-fill" style={{ width: percent }} />
      </div>
    </div>
  );
}

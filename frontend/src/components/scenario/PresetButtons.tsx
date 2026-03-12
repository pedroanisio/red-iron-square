import type { ReactElement } from "react";

const presets = [
  { key: "promotion", label: "The Promotion" },
  { key: "phone_call", label: "The Phone Call" },
  { key: "three_months", label: "Three Months Later" },
] as const;

/** Row of quick-pick buttons for the built-in scripted scenarios. */
export function PresetButtons({
  pending,
  onSelect,
}: {
  pending: boolean;
  onSelect: (scenarioKey: string) => void;
}): ReactElement {
  return (
    <div className="preset-grid">
      {presets.map((preset) => (
        <button
          key={preset.key}
          type="button"
          className="preset-button"
          disabled={pending}
          onClick={() => onSelect(preset.key)}
        >
          {preset.label}
        </button>
      ))}
    </div>
  );
}

import type { FormEvent, ReactElement } from "react";
import { useState } from "react";
import { PresetButtons } from "./PresetButtons";

/** Central control panel with preset buttons, a free-text scenario form, and a swap trigger. */
export function ScenarioPanel({
  scenario,
  pending,
  onPreset,
  onSubmit,
  onSwap,
}: {
  scenario: string;
  pending: boolean;
  onPreset: (scenarioKey: string) => void;
  onSubmit: (text: string) => Promise<boolean>;
  onSwap: () => void;
}): ReactElement {
  const [value, setValue] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (value.trim().length === 0) {
      return;
    }
    const wasSent = await onSubmit(value);
    if (wasSent) {
      setValue("");
    }
  }

  return (
    <section className="scenario-panel" aria-label="Scenario panel">
      <div className="scenario-frame">
        <p className="scenario-label">Current Situation</p>
        <p className="scenario-text">{scenario}</p>
      </div>
      <PresetButtons pending={pending} onSelect={onPreset} />
      <form className="scenario-form" onSubmit={handleSubmit}>
        <label htmlFor="custom-scenario">Open floor</label>
        <textarea
          id="custom-scenario"
          name="custom-scenario"
          value={value}
          placeholder="What if Luna runs into an old friend on the street?"
          onChange={(event) => setValue(event.target.value)}
          disabled={pending}
        />
        <div className="scenario-actions">
          <button type="submit" className="action-button" disabled={pending}>
            Send scenario
          </button>
          <button
            type="button"
            className="action-button action-button-ghost"
            disabled={pending}
            onClick={onSwap}
          >
            Swap personalities
          </button>
        </div>
      </form>
    </section>
  );
}

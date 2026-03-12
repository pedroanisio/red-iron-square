import type { ReactElement } from "react";
import { StateBar } from "./StateBar";
import { ThinkingState } from "../system/ThinkingState";
import type { DemoAgent } from "../../demo/types";

/** Card displaying an agent's identity, emotional state bars, and latest voice line. */
export function AgentCard({ agent }: { agent: DemoAgent }): ReactElement {
  const latestLine = agent.transcript.at(-1) ?? "Waiting for the moment to land.";
  return (
    <article className={`agent-card agent-card-${agent.key}`}>
      <div className="agent-topline">
        <div className="portrait-medallion" aria-hidden="true">
          {agent.name.slice(0, 1)}
        </div>
        <div>
          <p className="agent-nameplate">{agent.name}</p>
          <h2>{agent.name}</h2>
          <p className="agent-summary">{agent.summary}</p>
        </div>
      </div>
      <div className="emotion-pill">{agent.emotion_label}</div>
      <div className="state-stack">
        <StateBar label="Mood" value={agent.mood} tone="mood" />
        <StateBar label="Energy" value={agent.energy} tone="energy" />
        <StateBar label="Calm" value={agent.calm} tone="calm" />
      </div>
      <ThinkingState visible={agent.speaking} label={`${agent.name} is speaking`} />
      <blockquote className="agent-voice" aria-live="polite">
        {latestLine}
      </blockquote>
    </article>
  );
}

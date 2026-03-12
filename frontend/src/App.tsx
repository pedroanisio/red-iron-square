import type { ReactElement } from "react";
import { AgentCard } from "./components/agents/AgentCard";
import { DemoStage } from "./components/layout/DemoStage";
import { ScenarioPanel } from "./components/scenario/ScenarioPanel";
import { StatusBanner } from "./components/system/StatusBanner";
import { useDemoSession } from "./demo/useDemoSession";

/** Root component that wires the Two Minds demo session into the stage layout. */
function App(): ReactElement {
  const { state, runPreset, submitScenario, swapPersonalities } = useDemoSession();
  const luna = state.session?.agents.find((agent) => agent.key === "luna");
  const marco = state.session?.agents.find((agent) => agent.key === "marco");

  if (luna === undefined || marco === undefined) {
    return (
      <main className="stage-shell stage-shell-loading">
        <p className="eyebrow">Family-facing showcase</p>
        <h1>Two Minds Demo</h1>
        <p className="stage-subtitle">Preparing Luna, Marco, and the room.</p>
      </main>
    );
  }

  return (
    <DemoStage
      status={
        <StatusBanner
          line={state.statusLine}
          socketStatus={state.socketStatus}
          audioFallback={state.audioFallback}
          errorMessage={state.errorMessage}
        />
      }
      left={
        <section className="agent-column" aria-label="Luna panel">
          <AgentCard agent={luna} />
        </section>
      }
      center={
        <ScenarioPanel
          scenario={state.currentScenario}
          pending={state.pending}
          onPreset={runPreset}
          onSubmit={submitScenario}
          onSwap={swapPersonalities}
        />
      }
      right={
        <section className="agent-column" aria-label="Marco panel">
          <AgentCard agent={marco} />
        </section>
      }
    />
  );
}

export default App;

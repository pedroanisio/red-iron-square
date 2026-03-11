import type { ReactElement, ReactNode } from "react";

type DemoStageProps = {
  left: ReactNode;
  center: ReactNode;
  right: ReactNode;
  status: ReactNode;
};

export function DemoStage({
  left,
  center,
  right,
  status,
}: DemoStageProps): ReactElement {
  return (
    <main className="stage-shell">
      <header className="stage-marquee">
        <p className="eyebrow">Family-facing showcase</p>
        <h1>Two Minds Demo</h1>
        <p className="stage-subtitle">
          One room, one scenario, two people feeling it in completely different
          ways.
        </p>
        {status}
      </header>
      <section className="stage-grid">
        {left}
        {center}
        {right}
      </section>
    </main>
  );
}

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import App from "../App";
import { useDemoSession } from "../demo/useDemoSession";

vi.mock("../demo/useDemoSession", () => ({
  useDemoSession: vi.fn(),
}));

const mockedUseDemoSession = vi.mocked(useDemoSession);

describe("App", () => {
  it("renders the stage shell with both agents and the central controls", () => {
    mockedUseDemoSession.mockReturnValue({
      state: {
        session: {
          session_id: "demo-123",
          act_number: 1,
          turn_count: 1,
          agents: [
            {
              key: "luna",
              name: "Luna",
              summary: "Thoughtful. Cautious. Feels things deeply.",
              mood: -0.4,
              energy: 0.4,
              calm: 0.3,
              emotion_label: "Apprehension",
              transcript: ["That's a lot to take in."],
              speaking: false,
            },
            {
              key: "marco",
              name: "Marco",
              summary: "Curious. Bold. Bounces back fast.",
              mood: 0.5,
              energy: 0.7,
              calm: 0.6,
              emotion_label: "Excitement",
              transcript: ["I want to see where this goes."],
              speaking: true,
            },
          ],
        },
        currentScenario: "Offered a major career promotion.",
        statusLine: "Turn 1 complete.",
        pending: false,
        audioFallback: "Audio streaming not configured yet.",
        socketStatus: "open",
      },
      runPreset: vi.fn(),
      submitScenario: vi.fn(),
      swapPersonalities: vi.fn(),
    });

    const html = renderToStaticMarkup(<App />);

    expect(html).toContain("Two Minds Demo");
    expect(html).toContain("Luna panel");
    expect(html).toContain("Marco panel");
    expect(html).toContain("The Promotion");
    expect(html).toContain("Swap personalities");
    expect(html).toContain("Audio streaming not configured yet.");
  });
});

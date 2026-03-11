import { describe, expect, it } from "vitest";
import { demoReducer, initialDemoState } from "../demo/reducer";

const baseState = demoReducer(initialDemoState, {
  type: "sessionLoaded",
  session: {
    session_id: "demo-123",
    act_number: 1,
    turn_count: 0,
    agents: [
      {
        key: "luna",
        name: "Luna",
        summary: "Thoughtful. Cautious. Feels things deeply.",
        mood: 0,
        energy: 0.5,
        calm: 0.5,
        emotion_label: "Neutral",
        transcript: [],
        speaking: false,
      },
      {
        key: "marco",
        name: "Marco",
        summary: "Curious. Bold. Bounces back fast.",
        mood: 0,
        energy: 0.5,
        calm: 0.5,
        emotion_label: "Neutral",
        transcript: [],
        speaking: false,
      },
    ],
  },
});

describe("demoReducer", () => {
  it("reduces websocket turn events into ui state", () => {
    const afterScenario = demoReducer(baseState, {
      type: "event",
      event: {
        event_type: "scenario_received",
        session_id: "demo-123",
        payload: { description: "Offered a major career promotion." },
      },
    });
    const afterSnapshot = demoReducer(afterScenario, {
      type: "event",
      event: {
        event_type: "agent_state_updated",
        session_id: "demo-123",
        payload: {
          snapshot: {
            key: "luna",
            name: "Luna",
            summary: "Thoughtful. Cautious. Feels things deeply.",
            mood: -0.4,
            energy: 0.3,
            calm: 0.2,
            emotion_label: "Apprehension",
          },
        },
      },
    });
    const afterText = demoReducer(afterSnapshot, {
      type: "event",
      event: {
        event_type: "agent_text_started",
        session_id: "demo-123",
        payload: { agent_key: "luna", text: "That's a lot to take in." },
      },
    });
    const afterAudio = demoReducer(afterText, {
      type: "event",
      event: {
        event_type: "audio_unavailable",
        session_id: "demo-123",
        payload: { reason: "Audio streaming not configured yet." },
      },
    });
    const completed = demoReducer(afterAudio, {
      type: "event",
      event: {
        event_type: "turn_completed",
        session_id: "demo-123",
        payload: { turn_count: 1 },
      },
    });

    expect(completed.currentScenario).toContain("promotion");
    expect(completed.session?.agents[0].emotion_label).toBe("Apprehension");
    expect(completed.session?.agents[0].transcript).toContain(
      "That's a lot to take in.",
    );
    expect(completed.audioFallback).toBe("Audio streaming not configured yet.");
    expect(completed.pending).toBe(false);
  });

  it("resets transcripts and counters on swap", () => {
    const withTranscript = demoReducer(baseState, {
      type: "event",
      event: {
        event_type: "agent_text_started",
        session_id: "demo-123",
        payload: { agent_key: "luna", text: "Old line" },
      },
    });
    const swapped = demoReducer(withTranscript, {
      type: "event",
      event: {
        event_type: "swap_completed",
        session_id: "demo-123",
        payload: { swapped: true },
      },
    });

    expect(swapped.session?.turn_count).toBe(0);
    expect(swapped.session?.agents[0].transcript).toEqual([]);
    expect(swapped.statusLine).toMatch(/same faces, different inner lives/i);
  });
});

import type { DemoAgent, DemoEvent, DemoSession, DemoState } from "./types";

export const initialDemoState: DemoState = {
  session: null,
  currentScenario: "Choose a scripted moment or ask the room for a new one.",
  statusLine: "Initializing the stage.",
  pending: true,
  audioFallback: null,
  errorMessage: null,
  socketStatus: "idle",
};

type DemoAction =
  | { type: "sessionLoaded"; session: DemoSession }
  | { type: "sessionResynced"; session: DemoSession }
  | { type: "pending"; value: boolean }
  | { type: "socketStatus"; status: DemoState["socketStatus"] }
  | { type: "bootstrapFailed"; message: string }
  | { type: "requestFailed"; message: string }
  | { type: "reconnectStarted" }
  | { type: "event"; event: DemoEvent };

export function demoReducer(state: DemoState, action: DemoAction): DemoState {
  switch (action.type) {
    case "sessionLoaded":
      return {
        ...state,
        pending: false,
        errorMessage: null,
        session: {
          ...action.session,
          agents: action.session.agents.map(normalizeAgent),
        },
        statusLine: "Stage ready. Pick the first beat.",
      };
    case "sessionResynced":
      return {
        ...state,
        pending: false,
        errorMessage: null,
        session: {
          ...action.session,
          agents: action.session.agents.map(normalizeAgent),
        },
        statusLine: "Connection restored. The room is back in sync.",
      };
    case "pending":
      return { ...state, pending: action.value };
    case "socketStatus":
      return { ...state, socketStatus: action.status };
    case "bootstrapFailed":
      return {
        ...state,
        pending: false,
        socketStatus: "closed",
        errorMessage: action.message,
        statusLine: action.message,
      };
    case "requestFailed":
      return {
        ...state,
        pending: false,
        errorMessage: action.message,
        statusLine: action.message,
      };
    case "reconnectStarted":
      return {
        ...state,
        pending: false,
        errorMessage: null,
        statusLine: "Connection drifted. Rejoining the scene...",
        socketStatus: "connecting",
      };
    case "event":
      return reduceEvent(state, action.event);
    default:
      return state;
  }
}

function reduceEvent(state: DemoState, event: DemoEvent): DemoState {
  if (state.session === null) {
    return state;
  }
  if (event.event_type === "scenario_received") {
    return {
      ...state,
      currentScenario: String(event.payload.description ?? state.currentScenario),
      statusLine: "Both agents are taking it in.",
      pending: true,
      audioFallback: null,
      errorMessage: null,
    };
  }
  if (event.event_type === "scenario_enriched") {
      return {
        ...state,
        statusLine: `Scenario framed as ${String(event.payload.scenario_name ?? "custom")}.`,
        errorMessage: null,
      };
  }
  if (event.event_type === "agent_state_updated") {
    const snapshot = event.payload.snapshot as DemoAgent;
    return {
      ...state,
      session: {
        ...state.session,
        agents: state.session.agents.map((agent) =>
          agent.key === snapshot.key ? normalizeAgent(snapshot) : agent,
        ),
      },
    };
  }
  if (event.event_type === "agent_text_started") {
    return patchAgent(state, String(event.payload.agent_key), (agent) => ({
      ...agent,
      speaking: true,
      transcript: [...agent.transcript, String(event.payload.text ?? "")],
    }));
  }
  if (event.event_type === "agent_text_completed") {
    return patchAgent(state, String(event.payload.agent_key), (agent) => ({
      ...agent,
      speaking: false,
    }));
  }
  if (event.event_type === "audio_unavailable") {
    return {
      ...state,
      audioFallback: String(event.payload.reason ?? "Audio unavailable."),
    };
  }
  if (event.event_type === "turn_completed") {
      return {
        ...state,
        pending: false,
        errorMessage: null,
        statusLine: `Turn ${String(event.payload.turn_count)} complete.`,
        session: {
          ...state.session,
          turn_count: Number(event.payload.turn_count ?? state.session.turn_count),
        },
    };
  }
  if (event.event_type === "swap_completed") {
      return {
        ...state,
        pending: false,
        statusLine: "Personalities swapped. Same faces, different inner lives.",
        audioFallback: null,
        errorMessage: null,
        currentScenario: "Replay the opening scenario to feel the difference.",
        session: {
        ...state.session,
        turn_count: 0,
        agents: state.session.agents.map((agent) => ({
          ...agent,
          mood: 0,
          energy: 0.5,
          calm: 0.5,
          emotion_label: "Neutral",
          transcript: [],
          speaking: false,
        })),
      },
    };
  }
  return state;
}

function patchAgent(
  state: DemoState,
  agentKey: string,
  patch: (agent: DemoAgent) => DemoAgent,
): DemoState {
  if (state.session === null) {
    return state;
  }
  return {
    ...state,
    session: {
      ...state.session,
      agents: state.session.agents.map((agent) =>
        agent.key === agentKey ? patch(agent) : agent,
      ),
    },
  };
}

function normalizeAgent(agent: DemoAgent): DemoAgent {
  return {
    ...agent,
    transcript: agent.transcript ?? [],
    speaking: agent.speaking ?? false,
  };
}

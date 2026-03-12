export type DemoAgent = {
  key: "luna" | "marco";
  name: string;
  summary: string;
  mood: number;
  energy: number;
  calm: number;
  emotion_label: string;
  transcript: string[];
  speaking: boolean;
};

export type DemoSession = {
  session_id: string;
  act_number: number;
  turn_count: number;
  agents: DemoAgent[];
};

export type DemoEvent = {
  event_type:
    | "session_initialized"
    | "scenario_received"
    | "scenario_enriched"
    | "agent_state_updated"
    | "agent_text_started"
    | "agent_text_completed"
    | "audio_unavailable"
    | "turn_completed"
    | "swap_completed";
  session_id: string;
  payload: Record<string, unknown>;
};

export type DemoState = {
  session: DemoSession | null;
  currentScenario: string;
  statusLine: string;
  pending: boolean;
  audioFallback: string | null;
  errorMessage: string | null;
  socketStatus: "idle" | "connecting" | "open" | "closed";
};

export type DemoApiClient = {
  createSession: (signal?: AbortSignal) => Promise<DemoSession>;
  getSession: (sessionId: string, signal?: AbortSignal) => Promise<DemoSession>;
  runScripted: (sessionId: string, scenarioKey: string) => Promise<void>;
  runCustom: (sessionId: string, text: string) => Promise<void>;
  swap: (sessionId: string) => Promise<void>;
};

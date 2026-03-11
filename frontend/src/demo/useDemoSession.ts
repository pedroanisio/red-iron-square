import {
  startTransition,
  useEffect,
  useEffectEvent,
  useMemo,
  useReducer,
  useRef,
} from "react";
import { createDemoApiClient } from "./api";
import { createBrowserSocket, type DemoSocket, type SocketFactory } from "./socket";
import { demoReducer, initialDemoState } from "./reducer";
import type { DemoApiClient, DemoEvent } from "./types";

type Options = {
  api?: DemoApiClient;
  socketFactory?: SocketFactory;
};

export function useDemoSession(options: Options = {}) {
  const api = useMemo(() => options.api ?? createDemoApiClient(), [options.api]);
  const socketFactory = options.socketFactory ?? createBrowserSocket;
  const [state, dispatch] = useReducer(demoReducer, initialDemoState);
  const socketRef = useRef<DemoSocket | null>(null);

  const handleEvent = useEffectEvent((event: DemoEvent) => {
    startTransition(() => {
      dispatch({ type: "event", event });
    });
  });

  useEffect(() => {
    let active = true;
    dispatch({ type: "pending", value: true });
    dispatch({ type: "socketStatus", status: "connecting" });
    void api
      .createSession()
      .then((session) => {
        if (!active) {
          return;
        }
        dispatch({ type: "sessionLoaded", session });
        socketRef.current = socketFactory(
          session.session_id,
          handleEvent,
          (status) => dispatch({ type: "socketStatus", status }),
        );
      })
      .catch((error: unknown) => {
        if (!active) {
          return;
        }
        const message =
          error instanceof Error
            ? error.message
            : "Unable to reach the demo backend.";
        dispatch({ type: "bootstrapFailed", message });
      });
    return () => {
      active = false;
      socketRef.current?.close();
    };
  }, [api, socketFactory]);

  return {
    state,
    runPreset: async (scenarioKey: string) => {
      if (state.session === null) {
        return;
      }
      dispatch({ type: "pending", value: true });
      await api.runScripted(state.session.session_id, scenarioKey);
    },
    submitScenario: async (text: string) => {
      if (state.session === null || text.trim().length === 0) {
        return;
      }
      dispatch({ type: "pending", value: true });
      await api.runCustom(state.session.session_id, text.trim());
    },
    swapPersonalities: async () => {
      if (state.session === null) {
        return;
      }
      dispatch({ type: "pending", value: true });
      await api.swap(state.session.session_id);
    },
  };
}

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

let bootstrapSessionPromise: Promise<ReturnType<DemoApiClient["createSession"]> extends Promise<infer T> ? T : never> | null = null;

export function useDemoSession(options: Options = {}) {
  const api = useMemo(() => options.api ?? createDemoApiClient(), [options.api]);
  const socketFactory = options.socketFactory ?? createBrowserSocket;
  const [state, dispatch] = useReducer(demoReducer, initialDemoState);
  const socketRef = useRef<DemoSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const hadOpenSocketRef = useRef(false);
  const needsResyncRef = useRef(false);

  const handleEvent = useEffectEvent((event: DemoEvent) => {
    startTransition(() => {
      dispatch({ type: "event", event });
    });
  });

  const handleSocketStatus = useEffectEvent((status: "connecting" | "open" | "closed") => {
    dispatch({ type: "socketStatus", status });
    if (status === "open") {
      if (needsResyncRef.current && sessionIdRef.current !== null) {
        needsResyncRef.current = false;
        void api
          .getSession(sessionIdRef.current)
          .then((session) => {
            dispatch({ type: "sessionResynced", session });
          })
          .catch(() => {
            dispatch({
              type: "requestFailed",
              message: "Connection returned, but the stage could not be resynced yet.",
            });
          });
      }
      hadOpenSocketRef.current = true;
      return;
    }
    if (status === "connecting" && hadOpenSocketRef.current) {
      needsResyncRef.current = true;
      dispatch({ type: "reconnectStarted" });
    }
  });

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    dispatch({ type: "pending", value: true });
    dispatch({ type: "socketStatus", status: "connecting" });
    const createSessionPromise =
      bootstrapSessionPromise ?? api.createSession(controller.signal);
    bootstrapSessionPromise = createSessionPromise;
    void createSessionPromise
      .then((session) => {
        if (!active) {
          return;
        }
        sessionIdRef.current = session.session_id;
        dispatch({ type: "sessionLoaded", session });
        socketRef.current = socketFactory(
          session.session_id,
          handleEvent,
          handleSocketStatus,
        );
      })
      .catch((error: unknown) => {
        if (!active) {
          return;
        }
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        const message =
          error instanceof Error
            ? error.message
            : "Unable to reach the demo backend.";
        dispatch({ type: "bootstrapFailed", message });
      })
      .finally(() => {
        if (bootstrapSessionPromise === createSessionPromise) {
          bootstrapSessionPromise = null;
        }
      });
    return () => {
      active = false;
      controller.abort();
      if (bootstrapSessionPromise === createSessionPromise) {
        bootstrapSessionPromise = null;
      }
      sessionIdRef.current = null;
      socketRef.current?.close();
    };
  }, [api, socketFactory]);

  async function runAction(
    work: (sessionId: string) => Promise<void>,
    fallbackMessage: string,
  ): Promise<boolean> {
    if (state.session === null) {
      return false;
    }
    dispatch({ type: "pending", value: true });
    try {
      await work(state.session.session_id);
      return true;
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : fallbackMessage;
      dispatch({ type: "requestFailed", message });
      return false;
    }
  }

  return {
    state,
    runPreset: async (scenarioKey: string) => {
      await runAction(
        (sessionId) => api.runScripted(sessionId, scenarioKey),
        "Unable to run the scripted scenario.",
      );
    },
    submitScenario: async (text: string) => {
      if (text.trim().length === 0) {
        return false;
      }
      return runAction(
        (sessionId) => api.runCustom(sessionId, text.trim()),
        "Unable to send that scenario.",
      );
    },
    swapPersonalities: async () => {
      await runAction(
        (sessionId) => api.swap(sessionId),
        "Unable to swap personalities.",
      );
    },
  };
}

import type { DemoEvent } from "./types";

export type DemoSocket = {
  close: () => void;
};

export type SocketFactory = (
  sessionId: string,
  onEvent: (event: DemoEvent) => void,
  onStatus: (status: "open" | "closed") => void,
) => DemoSocket;

export const createBrowserSocket: SocketFactory = (
  sessionId,
  onEvent,
  onStatus,
) => {
  const configuredBase =
    import.meta.env.VITE_RED_IRON_SQUARE_WS_URL?.replace(/\/$/, "") ??
    (import.meta.env.DEV ? "ws://127.0.0.1:8000" : undefined);
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socketBase =
    configuredBase ?? `${protocol}://${window.location.host}`;
  const socket = new WebSocket(
    `${socketBase}/demo/sessions/${sessionId}/stream`,
  );
  socket.addEventListener("open", () => onStatus("open"));
  socket.addEventListener("close", () => onStatus("closed"));
  socket.addEventListener("message", (message) => {
    onEvent(JSON.parse(String(message.data)) as DemoEvent);
  });
  return {
    close() {
      socket.close();
    },
  };
};

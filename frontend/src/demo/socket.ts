import type { DemoEvent } from "./types";

export type DemoSocket = {
  close: () => void;
};

export type SocketFactory = (
  sessionId: string,
  onEvent: (event: DemoEvent) => void,
  onStatus: (status: "connecting" | "open" | "closed") => void,
) => DemoSocket;

export const createBrowserSocket: SocketFactory = (
  sessionId,
  onEvent,
  onStatus,
) => {
  let closedByClient = false;
  let reconnectTimer: number | null = null;
  let reconnectAttempt = 0;

  const configuredBase =
    import.meta.env.VITE_RED_IRON_SQUARE_WS_URL?.replace(/\/$/, "") ?? undefined;
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socketBase = configuredBase ?? `${protocol}://${window.location.host}`;

  function connect(): WebSocket {
    onStatus("connecting");
    const socket = new WebSocket(
      `${socketBase}/demo/sessions/${sessionId}/stream`,
    );
    socket.addEventListener("open", () => {
      reconnectAttempt = 0;
      onStatus("open");
    });
    socket.addEventListener("close", () => {
      if (closedByClient) {
        onStatus("closed");
        return;
      }
      onStatus("connecting");
      const delay = Math.min(1000 * 2 ** reconnectAttempt, 5000);
      reconnectAttempt += 1;
      reconnectTimer = window.setTimeout(() => {
        activeSocket = connect();
      }, delay);
    });
    socket.addEventListener("message", (message) => {
      onEvent(JSON.parse(String(message.data)) as DemoEvent);
    });
    return socket;
  }

  let activeSocket = connect();
  return {
    close() {
      closedByClient = true;
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }
      activeSocket.close();
    },
  };
};

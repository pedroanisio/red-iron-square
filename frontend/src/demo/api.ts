import type { DemoApiClient, DemoSession } from "./types";

type Envelope<T> = { data: T };
const API_BASE_URL =
  import.meta.env.VITE_RED_IRON_SQUARE_API_URL?.replace(/\/$/, "") ?? "";

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json() as Envelope<T>).data;
}

/**
 * Build a {@link DemoApiClient} bound to the given base URL.
 *
 * @param baseUrl - API root. Defaults to the `VITE_RED_IRON_SQUARE_API_URL` env var.
 * @returns A fully-wired client for session and scenario operations.
 */
export function createDemoApiClient(baseUrl = API_BASE_URL): DemoApiClient {
  return {
    createSession(signal) {
      return request<DemoSession>(`${baseUrl}/demo/sessions`, {
        method: "POST",
        signal,
        body: JSON.stringify({ act_number: 1 }),
      });
    },
    getSession(sessionId, signal) {
      return request<DemoSession>(`${baseUrl}/demo/sessions/${sessionId}`, {
        signal,
      });
    },
    runScripted(sessionId, scenarioKey) {
      return request(`${baseUrl}/demo/sessions/${sessionId}/scripted/${scenarioKey}`, {
        method: "POST",
      });
    },
    runCustom(sessionId, text) {
      return request(`${baseUrl}/demo/sessions/${sessionId}/scenarios`, {
        method: "POST",
        body: JSON.stringify({ text }),
      });
    },
    swap(sessionId) {
      return request(`${baseUrl}/demo/sessions/${sessionId}/swap`, {
        method: "POST",
      });
    },
  };
}

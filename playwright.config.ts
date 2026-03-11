import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  retries: 0,
  use: {
    baseURL: "http://127.0.0.1:5001",
    headless: true,
    colorScheme: "dark",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
  webServer: {
    command: "cd backend && uv run python -m src.ui.app",
    url: "http://127.0.0.1:5001",
    reuseExistingServer: false,
    timeout: 15_000,
    env: {
      FLASK_DEBUG: "0",
      RED_IRON_SQUARE_API_URL: "http://127.0.0.1:18999",
    },
  },
});

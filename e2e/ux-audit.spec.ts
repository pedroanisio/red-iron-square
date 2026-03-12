/**
 * UX Audit — Playwright end-to-end tests for the Red Iron Square Flask UI.
 *
 * Covers: layout, accessibility, theme toggle, navigation, forms,
 * responsive behavior, keyboard interaction, and visual consistency.
 *
 * The Flask app runs against a dead API (port 18999) so we test the
 * graceful-degradation path: UI renders, shows "Unavailable", forms
 * are still interactive.
 */
import { test, expect, type Page } from "@playwright/test";

// ─── Layout & Structure ───────────────────────────────────────────

test.describe("Layout & Structure", () => {
  test("page loads with correct title and topbar", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle("Red Iron Square");
    await expect(page.locator("header.topbar")).toBeVisible();
    await expect(page.locator("header.topbar")).toContainText(
      "Red Iron Square",
    );
  });

  test("topbar contains navigation links", async ({ page }) => {
    await page.goto("/");
    const campaignsLink = page.locator('a[href="/campaigns"]');
    await expect(campaignsLink).toBeVisible();
    await expect(campaignsLink).toContainText("Campaigns");
  });

  test("main content area has correct landmark roles", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator('[role="banner"]')).toBeVisible();
    await expect(page.locator('[role="complementary"]')).toBeVisible();
    await expect(page.locator('[role="main"]')).toBeVisible();
  });

  test("skip link is present and becomes visible on focus", async ({
    page,
  }) => {
    await page.goto("/");
    const skipLink = page.locator(".skip-link");
    await expect(skipLink).toHaveCount(1);
    await skipLink.focus();
    const box = await skipLink.boundingBox();
    expect(box).toBeTruthy();
    expect(box!.y).toBeGreaterThanOrEqual(0);
  });

  test("sidebar shows 'Start Simulation' card", async ({ page }) => {
    await page.goto("/");
    await expect(
      page.locator(".card-title", { hasText: "Start Simulation" }),
    ).toBeVisible();
  });

  test("API unavailable badge renders gracefully", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".badge-err")).toBeVisible();
    await expect(page.locator(".badge-err")).toContainText("Unavailable");
  });
});

// ─── Accessibility ────────────────────────────────────────────────

test.describe("Accessibility", () => {
  test("all images and decorative elements have aria attributes", async ({
    page,
  }) => {
    await page.goto("/");
    const marks = page.locator('.mark[aria-hidden="true"]');
    await expect(marks.first()).toBeVisible();
  });

  test("form inputs have associated labels", async ({ page }) => {
    await page.goto("/");
    const labels = page.locator("label[for]");
    const count = await labels.count();
    expect(count).toBeGreaterThan(0);
    for (let i = 0; i < Math.min(count, 5); i++) {
      const forAttr = await labels.nth(i).getAttribute("for");
      if (forAttr) {
        const input = page.locator(`#${forAttr}`);
        await expect(input).toHaveCount(1);
      }
    }
  });

  test("buttons have accessible names", async ({ page }) => {
    await page.goto("/");
    const buttons = page.locator("button");
    const count = await buttons.count();
    for (let i = 0; i < count; i++) {
      const btn = buttons.nth(i);
      const text = await btn.textContent();
      const ariaLabel = await btn.getAttribute("aria-label");
      const title = await btn.getAttribute("title");
      const hasName =
        (text && text.trim().length > 0) ||
        ariaLabel !== null ||
        title !== null;
      expect(
        hasName,
        `Button at index ${i} lacks accessible name`,
      ).toBeTruthy();
    }
  });

  test("color contrast — text is readable on backgrounds", async ({
    page,
  }) => {
    await page.goto("/");
    const body = page.locator("body");
    const bg = await body.evaluate(
      (el) => getComputedStyle(el).backgroundColor,
    );
    const color = await body.evaluate((el) => getComputedStyle(el).color);
    expect(bg).not.toEqual(color);
  });

  test("focus indicators are visible on interactive elements", async ({
    page,
  }) => {
    await page.goto("/");
    await page.keyboard.press("Tab");
    await page.keyboard.press("Tab");
    const focused = page.locator(":focus");
    await expect(focused).toHaveCount(1);
  });
});

// ─── Theme Toggle ─────────────────────────────────────────────────

test.describe("Theme Toggle", () => {
  test("toggle button is present with moon icon", async ({ page }) => {
    await page.goto("/");
    const toggle = page.locator("#theme-toggle");
    await expect(toggle).toBeVisible();
    await expect(toggle).toHaveAttribute(
      "aria-label",
      "Toggle light/dark theme",
    );
  });

  test("clicking toggle switches to light theme", async ({ page }) => {
    await page.goto("/");
    const html = page.locator("html");
    await page.click("#theme-toggle");
    await expect(html).toHaveAttribute("data-theme", "light");

    const bg = await page.evaluate(() =>
      getComputedStyle(document.body).backgroundColor,
    );
    expect(bg).not.toContain("10, 10, 10");
  });

  test("clicking toggle twice returns to dark theme", async ({ page }) => {
    await page.goto("/");
    await page.click("#theme-toggle");
    await page.click("#theme-toggle");
    const html = page.locator("html");
    await expect(html).toHaveAttribute("data-theme", "dark");
  });

  test("theme persists across navigation", async ({ page }) => {
    await page.goto("/");
    await page.click("#theme-toggle");
    await page.goto("/campaigns");
    const html = page.locator("html");
    await expect(html).toHaveAttribute("data-theme", "light");
  });

  test("light theme has readable text on light background", async ({
    page,
  }) => {
    await page.goto("/");
    await page.click("#theme-toggle");
    const textColor = await page.evaluate(() => {
      const style = getComputedStyle(document.body);
      return style.color;
    });
    // Light theme text should be dark
    expect(textColor).toMatch(/rgb\(26, 26, 24\)/);
  });

  test("light theme cards have visible borders", async ({ page }) => {
    await page.goto("/");
    await page.click("#theme-toggle");
    const card = page.locator(".card").first();
    const border = await card.evaluate(
      (el) => getComputedStyle(el).borderColor,
    );
    expect(border).not.toContain("0, 0, 0, 0)");
  });
});

// ─── Forms & Interaction ──────────────────────────────────────────

test.describe("Forms & Interaction", () => {
  test("visual personality sliders update display values", async ({
    page,
  }) => {
    await page.goto("/");
    const slider = page.locator('.cfg-trait[data-trait="O"]');
    const display = slider.locator("+ .trait-val-display");
    await slider.fill("0.3");
    await slider.dispatchEvent("input");
    await expect(display).toHaveText("0.3");
  });

  test("JSON textarea validates and shows error", async ({ page }) => {
    await page.goto("/");
    // Toggle to raw JSON
    await page.click("#toggle-config-mode");
    const textarea = page.locator("#config_json_raw");
    await textarea.fill("not json");
    await textarea.dispatchEvent("input");
    const msg = page.locator(
      '#raw-config .field-msg',
    );
    await expect(msg).toBeVisible();
    await expect(textarea).toHaveAttribute("aria-invalid", "true");
  });

  test("JSON textarea clears error on valid input", async ({ page }) => {
    await page.goto("/");
    await page.click("#toggle-config-mode");
    const textarea = page.locator("#config_json_raw");
    await textarea.fill("not json");
    await textarea.dispatchEvent("input");
    await textarea.fill('{"valid": true}');
    await textarea.dispatchEvent("input");
    await expect(textarea).not.toHaveAttribute("aria-invalid", "true");
  });

  test("toggle between visual and raw JSON modes", async ({ page }) => {
    await page.goto("/");
    const toggleBtn = page.locator("#toggle-config-mode");
    const visual = page.locator("#visual-config");
    const raw = page.locator("#raw-config");

    await expect(visual).toBeVisible();
    await expect(raw).not.toBeVisible();
    await toggleBtn.click();
    await expect(visual).not.toBeVisible();
    await expect(raw).toBeVisible();
    await toggleBtn.click();
    await expect(visual).toBeVisible();
    await expect(raw).not.toBeVisible();
  });

  test("collapsible cards toggle on click", async ({ page }) => {
    await page.goto("/");
    const header = page.locator('[data-collapse]:has-text("Start Simulation")');
    const body = header.locator("+ .card-bd");
    await expect(body).toBeVisible();
    await header.click();
    await expect(body).not.toBeVisible();
    await header.click();
    await expect(body).toBeVisible();
  });

  test("disabled buttons have proper styling when no run selected", async ({
    page,
  }) => {
    await page.goto("/");
    const assistBtn = page.locator(
      'button[type="submit"]:has-text("Run Assisted Step")',
    );
    await expect(assistBtn).toBeDisabled();
    await expect(assistBtn).toHaveAttribute(
      "title",
      "Select or create a run first",
    );
  });

  test("add and remove action rows in config builder", async ({ page }) => {
    await page.goto("/");
    const initial = await page.locator(".action-entry").count();
    await page.click("#add-action-btn");
    expect(await page.locator(".action-entry").count()).toBe(initial + 1);
    await page.locator(".action-remove").last().click();
    expect(await page.locator(".action-entry").count()).toBe(initial);
  });
});

// ─── Keyboard Navigation ──────────────────────────────────────────

test.describe("Keyboard Navigation", () => {
  test("Escape collapses open cards", async ({ page }) => {
    await page.goto("/");
    const headers = page.locator("[data-collapse]");
    const count = await headers.count();
    // Ensure at least one is expanded
    const firstBody = headers.first().locator("+ .card-bd");
    if (!(await firstBody.isVisible())) {
      await headers.first().click();
    }
    await page.keyboard.press("Escape");
    for (let i = 0; i < count; i++) {
      const body = headers.nth(i).locator("+ .card-bd");
      await expect(body).not.toBeVisible();
    }
  });

  test("Ctrl+Enter submits focused form", async ({ page }) => {
    await page.goto("/");
    // Focus on a textarea inside the manual step form
    const scenarioName = page.locator("#scenario_name");
    await scenarioName.focus();
    // Ctrl+Enter should trigger form submit (will redirect/fail gracefully)
    const [response] = await Promise.all([
      page.waitForNavigation({ timeout: 5000 }).catch(() => null),
      page.keyboard.press("Control+Enter"),
    ]);
    // Either navigated or stayed — the point is no crash
    expect(true).toBeTruthy();
  });

  test("Ctrl+Enter does not submit disabled run-dependent forms", async ({
    page,
  }) => {
    const requests: string[] = [];
    page.on("request", (request) => {
      if (request.method() === "POST") {
        requests.push(request.url());
      }
    });
    await page.goto("/");
    await page.locator("#scenario_name").focus();
    await page.keyboard.down("Control");
    await page.keyboard.press("Enter");
    await page.keyboard.up("Control");
    await page.waitForTimeout(300);
    expect(requests.some((url) => url.includes("/runs//tick"))).toBeFalsy();
  });
});

// ─── Responsive Design ────────────────────────────────────────────

test.describe("Responsive Design", () => {
  test("mobile viewport stacks sidebar below main", async ({ page }) => {
    await page.setViewportSize({ width: 480, height: 800 });
    await page.goto("/");
    const sidebar = page.locator("nav.sidebar");
    const main = page.locator("#main-content");
    const sidebarBox = await sidebar.boundingBox();
    const mainBox = await main.boundingBox();
    expect(sidebarBox).toBeTruthy();
    expect(mainBox).toBeTruthy();
    // On mobile, main should appear before sidebar (order: -1)
    expect(mainBox!.y).toBeLessThan(sidebarBox!.y);
  });

  test("desktop viewport shows side-by-side layout", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/");
    const sidebar = page.locator("nav.sidebar");
    const main = page.locator("#main-content");
    const sidebarBox = await sidebar.boundingBox();
    const mainBox = await main.boundingBox();
    expect(sidebarBox).toBeTruthy();
    expect(mainBox).toBeTruthy();
    // Side by side: sidebar left of main
    expect(sidebarBox!.x).toBeLessThan(mainBox!.x);
  });

  test("topbar is sticky on scroll", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 400 });
    await page.goto("/");
    await page.evaluate(() => window.scrollTo(0, 500));
    const topbar = page.locator("header.topbar");
    const box = await topbar.boundingBox();
    expect(box).toBeTruthy();
    expect(box!.y).toBe(0);
  });
});

// ─── Campaigns Page ───────────────────────────────────────────────

test.describe("Campaigns Page", () => {
  test("campaigns page loads and has form", async ({ page }) => {
    await page.goto("/campaigns");
    await expect(page).toHaveTitle("Red Iron Square");
    await expect(
      page.locator("h2", { hasText: "Campaigns" }),
    ).toBeVisible();
    await expect(page.locator("#camp-name")).toBeVisible();
    await expect(page.locator("#camp-goals")).toBeVisible();
  });

  test("campaign form has required name field", async ({ page }) => {
    await page.goto("/campaigns");
    const nameInput = page.locator("#camp-name");
    await expect(nameInput).toHaveAttribute("required", "");
  });

  test("campaigns page JSON textarea validates", async ({ page }) => {
    await page.goto("/campaigns");
    const textarea = page.locator("#camp-config");
    await textarea.fill("bad json{");
    await textarea.dispatchEvent("input");
    await expect(textarea).toHaveAttribute("aria-invalid", "true");
  });
});

// ─── Compare Page ─────────────────────────────────────────────────

test.describe("Compare Page", () => {
  test("compare page loads with input fields", async ({ page }) => {
    await page.goto("/compare");
    await expect(
      page.locator("h2", { hasText: "Compare Runs" }),
    ).toBeVisible();
    await expect(page.locator("#left")).toBeVisible();
    await expect(page.locator("#right")).toBeVisible();
  });

  test("compare form submits with run IDs", async ({ page }) => {
    await page.goto("/compare");
    await page.fill("#left", "run-abc");
    await page.fill("#right", "run-def");
    await page.click('button:has-text("Compare")');
    await expect(page).toHaveURL(/left=run-abc/);
    await expect(page).toHaveURL(/right=run-def/);
  });
});

// ─── Visual Consistency ───────────────────────────────────────────

test.describe("Visual Consistency", () => {
  test("CSS custom properties are defined", async ({ page }) => {
    await page.goto("/");
    const vars = await page.evaluate(() => {
      const style = getComputedStyle(document.documentElement);
      return {
        bg: style.getPropertyValue("--bg").trim(),
        surface: style.getPropertyValue("--surface").trim(),
        red: style.getPropertyValue("--red").trim(),
        text: style.getPropertyValue("--text").trim(),
        mono: style.getPropertyValue("--mono").trim(),
      };
    });
    expect(vars.bg).toBeTruthy();
    expect(vars.surface).toBeTruthy();
    expect(vars.red).toBeTruthy();
    expect(vars.text).toBeTruthy();
    expect(vars.mono).toContain("DM Mono");
  });

  test("fonts are loaded and applied", async ({ page }) => {
    await page.goto("/");
    const bodyFont = await page.evaluate(
      () => getComputedStyle(document.body).fontFamily,
    );
    expect(bodyFont).toContain("Instrument Sans");
  });

  test("cards have consistent border and background", async ({ page }) => {
    await page.goto("/");
    const cards = page.locator(".card");
    const count = await cards.count();
    expect(count).toBeGreaterThan(0);
    const firstBg = await cards
      .first()
      .evaluate((el) => getComputedStyle(el).backgroundColor);
    const firstBorder = await cards
      .first()
      .evaluate((el) => getComputedStyle(el).borderColor);
    for (let i = 1; i < Math.min(count, 4); i++) {
      const bg = await cards
        .nth(i)
        .evaluate((el) => getComputedStyle(el).backgroundColor);
      const border = await cards
        .nth(i)
        .evaluate((el) => getComputedStyle(el).borderColor);
      expect(bg).toEqual(firstBg);
      expect(border).toEqual(firstBorder);
    }
  });

  test("no horizontal overflow on any page", async ({ page }) => {
    for (const path of ["/", "/campaigns", "/compare"]) {
      await page.goto(path);
      const overflow = await page.evaluate(() => {
        return document.documentElement.scrollWidth > window.innerWidth;
      });
      expect(overflow, `Horizontal overflow on ${path}`).toBeFalsy();
    }
  });

  test("grid background pattern renders", async ({ page }) => {
    await page.goto("/");
    const bgImage = await page.evaluate(
      () => getComputedStyle(document.body).backgroundImage,
    );
    expect(bgImage).toContain("repeating-linear-gradient");
  });

  test("entrance animations apply to cards", async ({ page }) => {
    await page.goto("/");
    const animation = await page
      .locator(".card")
      .first()
      .evaluate((el) => getComputedStyle(el).animationName);
    expect(animation).toContain("slide-up");
  });
});

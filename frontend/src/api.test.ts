import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  formatApiDetail,
  REQUEST_TIMEOUT_MS,
  sendChat
} from "./api";

describe("API errors", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("formats FastAPI validation arrays without object stringification", () => {
    const detail = formatApiDetail([
      { type: "missing", loc: ["body", "message"], msg: "Field required" },
      { type: "value_error", loc: ["body", "locale"], msg: "Invalid locale" }
    ], "请求失败");

    expect(detail).toBe("Field required（message）；Invalid locale（locale）");
    expect(detail).not.toContain("[object Object]");
  });

  it("uses the formatted validation detail in rejected requests", async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      detail: [{ loc: ["body", "message"], msg: "Field required", type: "missing" }]
    }), {
      status: 422,
      headers: { "Content-Type": "application/json" }
    }));

    await expect(sendChat("")).rejects.toEqual(
      new ApiError("Field required（message）", 422)
    );
  });

  it("waits slightly longer than the proxy timeout", () => {
    expect(REQUEST_TIMEOUT_MS).toBeGreaterThan(600_000);
  });
});

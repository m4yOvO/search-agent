import type { ChatResponse, GraphPayload } from "./types";

// A prompt-driven turn can require several sequential model/tool decisions. Nginx
// allows 600s for /chat; keep the browser slightly beyond that boundary so it does
// not abort a request that is still running and invite a duplicate retry.
export const REQUEST_TIMEOUT_MS = 615_000;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number
  ) {
    super(message);
    this.name = "ApiError";
  }
}

interface ValidationIssue {
  loc?: unknown;
  msg?: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Convert FastAPI's validation issue arrays into readable, non-object UI copy. */
export function formatApiDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (!Array.isArray(detail)) return fallback;

  const issues = detail.flatMap((item): string[] => {
    if (typeof item === "string" && item.trim()) return [item];
    if (!isRecord(item)) return [];

    const issue = item as ValidationIssue;
    if (typeof issue.msg !== "string" || !issue.msg.trim()) return [];
    const location = Array.isArray(issue.loc)
      ? issue.loc
        .filter((part) => part !== "body")
        .map(String)
        .join(".")
      : "";
    return [location ? `${issue.msg}（${location}）` : issue.msg];
  });

  return issues.length ? issues.join("；") : fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(path, { ...init, signal: controller.signal });
    if (!response.ok) {
      let detail = `请求失败（${response.status}）`;
      try {
        const body = (await response.json()) as { detail?: unknown };
        detail = formatApiDetail(body.detail, detail);
      } catch {
        // Keep the status-based message for non-JSON failures.
      }
      throw new ApiError(detail, response.status);
    }
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("查询超时，请稍后重试。", 408);
    }
    throw new ApiError("无法连接到探索服务，请检查服务状态后重试。");
  } finally {
    window.clearTimeout(timeout);
  }
}

export function sendChat(
  message: string,
  conversationId?: string
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: conversationId,
      message,
      locale: "zh-CN"
    })
  });
}

export function getGraph(conversationId: string): Promise<GraphPayload> {
  const query = new URLSearchParams({ conversation_id: conversationId });
  return request<GraphPayload>(`/graph?${query.toString()}`);
}

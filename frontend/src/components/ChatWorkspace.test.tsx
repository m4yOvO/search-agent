import { fireEvent, render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ChatMessage } from "../types";
import { ChatWorkspace, isNearMessageBottom } from "./ChatWorkspace";

function message(id: string): ChatMessage {
  return {
    id,
    role: "assistant",
    content: `回答 ${id}`,
    createdAt: new Date("2026-07-17T00:00:00Z")
  };
}

function workspace(messages: ChatMessage[]) {
  return (
    <ChatWorkspace
      messages={messages}
      isLoading={false}
      canRetry={false}
      onSend={vi.fn()}
      onRetry={vi.fn()}
      onReset={vi.fn()}
    />
  );
}

describe("ChatWorkspace scrolling", () => {
  it("classifies only streams close to the bottom as followable", () => {
    expect(isNearMessageBottom({ scrollTop: 720, clientHeight: 220, scrollHeight: 1000 })).toBe(true);
    expect(isNearMessageBottom({ scrollTop: 300, clientHeight: 220, scrollHeight: 1000 })).toBe(false);
  });

  it("keeps a reader's position unless they were already near the bottom", () => {
    const { container, rerender } = render(workspace([message("1")]));
    const stream = container.querySelector<HTMLDivElement>(".message-stream");
    expect(stream).not.toBeNull();
    if (!stream) return;

    Object.defineProperties(stream, {
      scrollHeight: { configurable: true, value: 1000 },
      clientHeight: { configurable: true, value: 220 },
      scrollTop: { configurable: true, writable: true, value: 300 }
    });
    fireEvent.scroll(stream);
    rerender(workspace([message("1"), message("2")]));
    expect(stream.scrollTop).toBe(300);

    stream.scrollTop = 730;
    fireEvent.scroll(stream);
    rerender(workspace([message("1"), message("2"), message("3")]));
    expect(stream.scrollTop).toBe(1000);
  });
});

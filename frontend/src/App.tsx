import { useCallback, useMemo, useState } from "react";

import { ApiError, sendChat } from "./api";
import { ChatWorkspace } from "./components/ChatWorkspace";
import { GraphWorkspace } from "./components/GraphWorkspace";
import type { ChatMessage, GraphPayload } from "./types";

function emptyGraph(): GraphPayload {
  return {
    graph_id: "graph:empty",
    nodes: [],
    edges: [],
    evidence: [],
    generated_at: new Date().toISOString(),
    data_version: "demo-not-loaded"
  };
}

function messageId(role: "user" | "assistant"): string {
  return `${role}:${crypto.randomUUID()}`;
}

export default function App() {
  const [conversationId, setConversationId] = useState<string>();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [graph, setGraph] = useState<GraphPayload>(() => emptyGraph());
  const [isLoading, setIsLoading] = useState(false);
  const [retryQuery, setRetryQuery] = useState<string | null>(null);

  const runQuery = useCallback(async (query: string, appendUser = true) => {
    if (isLoading) return;
    if (appendUser) {
      setMessages((current) => [
        ...current,
        { id: messageId("user"), role: "user", content: query, createdAt: new Date() }
      ]);
    } else {
      setMessages((current) => current.filter((message) => !message.failed));
    }
    setIsLoading(true);
    setRetryQuery(null);

    try {
      const response = await sendChat(query, conversationId);
      const responseFailed = response.status === "failed";
      setConversationId(response.conversation_id);
      setGraph(response.graph);
      if (responseFailed) setRetryQuery(query);
      setMessages((current) => [
        ...current,
        {
          id: messageId("assistant"),
          role: "assistant",
          content: response.answer,
          memory: response.memory,
          trace: response.trace,
          createdAt: new Date(),
          failed: responseFailed
        }
      ]);
    } catch (error) {
      const content = error instanceof ApiError ? error.message : "查询失败，请稍后重试。";
      setRetryQuery(query);
      setMessages((current) => [
        ...current,
        {
          id: messageId("assistant"),
          role: "assistant",
          content,
          createdAt: new Date(),
          failed: true
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [conversationId, isLoading]);

  const retry = useCallback(() => {
    if (retryQuery) void runQuery(retryQuery, false);
  }, [retryQuery, runQuery]);

  const reset = useCallback(() => {
    setConversationId(undefined);
    setMessages([]);
    setGraph(emptyGraph());
    setRetryQuery(null);
  }, []);

  const sessionLabel = useMemo(
    () => conversationId ? `会话 ${conversationId.slice(0, 8)}` : "等待开始",
    [conversationId]
  );

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="关系镜首页">
          <span className="brand-mark" aria-hidden="true">关</span>
          <span><strong>关系镜</strong><small>RELATION LENS</small></span>
        </a>
        <div className="system-status">
          <span className="status-dot" aria-hidden="true" />
          <span>{sessionLabel}</span>
          <span className="header-divider" aria-hidden="true" />
          <span>本地演示数据</span>
        </div>
      </header>

      <main id="top" className="workspace-grid">
        <ChatWorkspace
          messages={messages}
          isLoading={isLoading}
          canRetry={retryQuery != null}
          onSend={(query) => void runQuery(query)}
          onRetry={retry}
          onReset={reset}
        />
        <GraphWorkspace graph={graph} />
      </main>

      <footer className="disclosure">
        <span>DEMO / PUBLIC FIXTURES</span>
        <p>结果来自本地演示数据，不代表实时工商或法律结论。请勿用于投资、法律或尽调决策。</p>
      </footer>
    </div>
  );
}

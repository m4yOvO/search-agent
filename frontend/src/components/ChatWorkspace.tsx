import {
  FormEvent,
  KeyboardEvent,
  UIEvent,
  useLayoutEffect,
  useRef,
  useState
} from "react";

import type { ChatMessage } from "../types";
import { CacheBadge } from "./CacheBadge";

const SUGGESTIONS = [
  "马云创办了哪些公司？",
  "阿里巴巴集团拥有哪些公司？",
  "这些公司在哪？"
];

interface ChatWorkspaceProps {
  messages: ChatMessage[];
  isLoading: boolean;
  canRetry: boolean;
  onSend: (message: string) => void;
  onRetry: () => void;
  onReset: () => void;
}

interface ScrollMetrics {
  scrollTop: number;
  clientHeight: number;
  scrollHeight: number;
}

export function isNearMessageBottom(metrics: ScrollMetrics, threshold = 72): boolean {
  return metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight <= threshold;
}

export function ChatWorkspace({
  messages,
  isLoading,
  canRetry,
  onSend,
  onRetry,
  onReset
}: ChatWorkspaceProps) {
  const [draft, setDraft] = useState("");
  const streamRef = useRef<HTMLDivElement | null>(null);
  const wasNearBottomRef = useRef(true);
  const trimmed = draft.trim();

  useLayoutEffect(() => {
    const stream = streamRef.current;
    if (!stream || !wasNearBottomRef.current) return;
    stream.scrollTop = stream.scrollHeight;
  }, [messages.length, isLoading]);

  function submit(event?: FormEvent) {
    event?.preventDefault();
    if (!trimmed || isLoading) return;
    onSend(trimmed);
    setDraft("");
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.nativeEvent.isComposing || event.keyCode === 229) return;
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  function onMessageScroll(event: UIEvent<HTMLDivElement>) {
    wasNearBottomRef.current = isNearMessageBottom(event.currentTarget);
  }

  return (
    <section className="chat-workspace" aria-label="对话探索">
      <header className="chat-heading">
        <div>
          <p className="eyebrow">自然语言研究</p>
          <h1>从一个名字，追到一张关系网</h1>
        </div>
        <button className="text-button" type="button" onClick={onReset} disabled={isLoading}>
          新会话
        </button>
      </header>

      <div
        ref={streamRef}
        className="message-stream"
        aria-live="polite"
        aria-busy={isLoading}
        onScroll={onMessageScroll}
      >
        {messages.length === 0 ? (
          <div className="chat-empty">
            <p>
              试着查询原始数据中的人物—企业关系，再继续追问地点。系统会展示每一条演示关系，
              并在可安全复用时写入长期记忆。
            </p>
            <div className="prompt-list" aria-label="示例问题">
              {SUGGESTIONS.slice(0, 2).map((suggestion, index) => (
                <button
                  key={suggestion}
                  type="button"
                  aria-label={suggestion}
                  onClick={() => onSend(suggestion)}
                  disabled={isLoading}
                >
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  {suggestion}
                  <span aria-hidden="true">↗</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <article
              key={message.id}
              className={`message message--${message.role}${message.failed ? " message--failed" : ""}`}
            >
              <div className="message-meta">
                <span>{message.role === "user" ? "你" : "关系研究员"}</span>
                <time dateTime={message.createdAt.toISOString()}>
                  {message.createdAt.toLocaleTimeString("zh-CN", {
                    hour: "2-digit",
                    minute: "2-digit"
                  })}
                </time>
              </div>
              <p>{message.content}</p>
              {message.memory || message.trace ? (
                <div className="message-badges">
                  {message.memory ? <CacheBadge memory={message.memory} /> : null}
                  {message.trace ? (
                    <span
                      className="model-badge"
                      title={`Planner ${message.trace.planner_model_calls} · Researcher ${message.trace.researcher_model_calls} · Visualizer ${message.trace.visualizer_model_calls}`}
                    >
                      <span aria-hidden="true">◇</span>
                      {message.trace.model_calls === 0
                        ? "缓存直返 · 0 模型调用"
                        : `${message.trace.model_name ?? message.trace.model_provider} · ${message.trace.model_calls} 次模型调用`}
                    </span>
                  ) : null}
                </div>
              ) : null}
              {message.failed && canRetry ? (
                <button className="retry-button" type="button" onClick={onRetry} disabled={isLoading}>
                  重试这条查询
                </button>
              ) : null}
            </article>
          ))
        )}

        {isLoading ? (
          <div className="researching" role="status">
            <span className="researching-dot" />
            <span>Planner 正在拆解问题，Researcher 正在核对关系…</span>
          </div>
        ) : null}
      </div>

      <form className="composer" onSubmit={submit}>
        <label htmlFor="query-composer">继续探索</label>
        <div className="composer-field">
          <textarea
            id="query-composer"
            value={draft}
            onChange={(event) => setDraft(event.target.value.slice(0, 1000))}
            onKeyDown={onComposerKeyDown}
            placeholder={messages.length ? "例如：这些公司在哪？" : "输入人物、企业或关系问题…"}
            rows={2}
            maxLength={1000}
            disabled={isLoading}
          />
          <button type="submit" disabled={!trimmed || isLoading} aria-label="发送查询">
            {isLoading ? "核对中" : "发送"}
            <span aria-hidden="true">↗</span>
          </button>
        </div>
        <div className="composer-notes">
          <span>Enter 发送 · Shift + Enter 换行</span>
          <span>{draft.length}/1000</span>
        </div>
      </form>
    </section>
  );
}

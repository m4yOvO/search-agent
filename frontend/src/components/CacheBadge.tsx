import type { MemoryMetadata } from "../types";

interface CacheBadgeProps {
  memory: MemoryMetadata;
}

export function CacheBadge({ memory }: CacheBadgeProps) {
  if (memory.cache_hit) {
    return (
      <span className="cache-badge cache-badge--hit" title={memory.match_type ?? "长期缓存命中"}>
        <span aria-hidden="true">●</span>
        长期记忆命中 · {memory.status === "hot" ? "HOT" : "WARM"}
      </span>
    );
  }

  if (memory.status === "warm") {
    return (
      <span className="cache-badge cache-badge--warm">
        <span aria-hidden="true">○</span>
        已写入长期记忆 · WARM
      </span>
    );
  }

  return (
    <span className="cache-badge cache-badge--skip">
      <span aria-hidden="true">—</span>
      本轮未复用缓存
    </span>
  );
}

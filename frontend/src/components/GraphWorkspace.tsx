import { lazy, Suspense, useEffect, useMemo, useState } from "react";

import { relationDisplayLabel } from "../relationPresentation";
import type { GraphNode, GraphPayload, NodeType } from "../types";

const InteractiveGraph = lazy(() =>
  import("./InteractiveGraph").then((module) => ({ default: module.InteractiveGraph }))
);

const TYPE_LABELS: Record<NodeType, string> = {
  person: "人物",
  company: "企业",
  location: "地点"
};

function propertyLabel(key: string): string {
  const labels: Record<string, string> = {
    founded_year: "成立年份",
    nationality: "国籍",
    country: "国家",
    summary: "说明",
    source_id: "来源记录",
    legal_rep_id: "关联人物",
    location_id: "地点索引",
    demo_data: "数据类型"
  };
  return labels[key] ?? key.replaceAll("_", " ");
}

function propertyValue(value: unknown): string {
  if (Array.isArray(value)) return value.join("、");
  if (typeof value === "boolean") return value ? "本地演示数据" : "否";
  if (value == null) return "—";
  return String(value);
}

interface GraphWorkspaceProps {
  graph: GraphPayload;
}

export function GraphWorkspace({ graph }: GraphWorkspaceProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = useMemo(
    () => graph.nodes.find((node) => node.id === selectedId) ?? null,
    [graph.nodes, selectedId]
  );

  useEffect(() => {
    if (selectedId && !graph.nodes.some((node) => node.id === selectedId)) {
      setSelectedId(null);
    }
  }, [graph.nodes, selectedId]);

  return (
    <section className="graph-workspace" aria-label="企业关系图谱">
      <header className="graph-heading">
        <div>
          <p className="eyebrow">证据化关系图谱</p>
          <h2>当前会话视图</h2>
        </div>
        <div className="graph-metrics" aria-label="图谱统计">
          <span><strong>{graph.nodes.length}</strong> 节点</span>
          <span><strong>{graph.edges.length}</strong> 关系</span>
        </div>
      </header>

      <div className="graph-stage">
        <div className="legend" aria-label="图例">
          <span><i className="legend-shape legend-shape--person" />人物</span>
          <span><i className="legend-shape legend-shape--company" />企业</span>
          <span><i className="legend-shape legend-shape--location" />地点</span>
        </div>
        {graph.nodes.length > 0 ? (
          <Suspense fallback={<div className="graph-loading" role="status">图谱引擎加载中…</div>}>
            <InteractiveGraph
              graph={graph}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          </Suspense>
        ) : (
          <div className="graph-canvas-shell">
            <div
              className="graph-canvas"
              role="img"
              aria-label="企业关系图，当前暂无节点或关系。"
            />
            <div className="graph-empty" aria-hidden="true">
              <span className="graph-empty-mark">◎</span>
              <p>关系图等待第一条查询</p>
              <small>回答中的人物、企业与地点会在这里连接起来</small>
            </div>
          </div>
        )}
        <EntityInspector graph={graph} node={selected} onSelect={setSelectedId} />
      </div>
    </section>
  );
}

function EntityInspector({
  graph,
  node,
  onSelect
}: {
  graph: GraphPayload;
  node: GraphNode | null;
  onSelect: (nodeId: string) => void;
}) {
  const relatedEdges = node
    ? graph.edges.filter((edge) => edge.source === node.id || edge.target === node.id)
    : [];
  const selectedEvidenceIds = new Set([
    ...(node?.evidence_ids ?? []),
    ...relatedEdges.flatMap((edge) => edge.evidence_ids)
  ]);
  const selectedEvidence = graph.evidence.filter((item) => selectedEvidenceIds.has(item.id));

  return (
    <aside className="entity-inspector" aria-live="polite">
      {node ? (
        <>
          <div className="inspector-title">
            <span className={`entity-type entity-type--${node.type}`}>{TYPE_LABELS[node.type]}</span>
            <h3>{node.label}</h3>
            <code>{node.id}</code>
          </div>
          <dl className="property-list">
            {Object.entries(node.properties)
              .filter(([key]) => key !== "aliases")
              .map(([key, value]) => (
                <div key={key}>
                  <dt>{propertyLabel(key)}</dt>
                  <dd>{propertyValue(value)}</dd>
                </div>
              ))}
          </dl>
          <div className="inspector-section">
            <h4>来源与更新时间</h4>
            <p>{relatedEdges.length} 条当前关系 · {selectedEvidence.length} 条相关证据</p>
            <ul className="evidence-records">
              {selectedEvidence.map((evidence) => (
                <li key={evidence.id}>
                  <strong>{evidence.provider}</strong>
                  <span>{evidence.source_kind} · 记录 {evidence.record_id}</span>
                  <time dateTime={evidence.updated_at}>
                    数据更新 {new Date(evidence.updated_at).toLocaleDateString("zh-CN")}
                  </time>
                  {evidence.source_url ? (
                    <a href={evidence.source_url} target="_blank" rel="noreferrer">打开公开来源</a>
                  ) : null}
                </li>
              ))}
            </ul>
          </div>
        </>
      ) : (
        <div className="inspector-empty">
          <p className="eyebrow">节点检查器</p>
          <h3>选择一个实体</h3>
          <p>点击图中的人物、企业或地点，查看属性、关系数量和证据索引。</p>
          <p className="dataset-stamp">
            数据版本 <code>{graph.data_version}</code><br />
            图谱更新 {new Date(graph.generated_at).toLocaleString("zh-CN")}
          </p>
        </div>
      )}

      <details className="accessible-graph">
        <summary>查看图谱文字列表</summary>
        <div>
          <h4>节点</h4>
          <ul>
            {graph.nodes.length ? graph.nodes.map((item) => (
              <li key={item.id}>
                <button
                  className="accessible-node-button"
                  type="button"
                  aria-current={node?.id === item.id ? "true" : undefined}
                  aria-label={`选择${TYPE_LABELS[item.type]}：${item.label}`}
                  onClick={() => onSelect(item.id)}
                >
                  <span>{TYPE_LABELS[item.type]}</span>
                  {item.label}
                </button>
              </li>
            )) : <li>暂无节点</li>}
          </ul>
          <h4>关系</h4>
          <ul>
            {graph.edges.length ? graph.edges.map((edge) => {
              const source = graph.nodes.find((item) => item.id === edge.source)?.label ?? edge.source;
              const target = graph.nodes.find((item) => item.id === edge.target)?.label ?? edge.target;
              return <li key={edge.id}>{source} — {relationDisplayLabel(edge)} → {target}</li>;
            }) : <li>暂无关系</li>}
          </ul>
        </div>
      </details>
    </aside>
  );
}

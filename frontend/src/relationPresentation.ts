import type { GraphEdge, RelationType } from "./types";

/**
 * Presentation-only labels for the source vocabulary in `relations 1.json`.
 * The raw relation remains untouched on GraphEdge.label/properties so evidence
 * inspection and API round-tripping never depend on localized UI copy.
 */
const RAW_RELATION_LABELS: Readonly<Record<string, string>> = {
  CEO_of: "担任 CEO",
  Chairman_of: "担任董事长",
  Chairwoman_of: "担任董事长",
  "Co-founder_of": "联合创办",
  Competes_with: "竞争",
  Former_CEO_of: "曾任 CEO",
  Former_Chairman_of: "曾任董事长",
  Former_President_of: "曾任总裁",
  Founder_of: "创办",
  Headquartered_in: "总部位于",
  Invested_in: "投资",
  Owns: "持有",
  Partner_with: "合作",
  Supplier_to: "供应",
  Uses_AI_from: "使用其 AI"
};

const TYPED_RELATION_LABELS: Readonly<Record<RelationType, string>> = {
  controls: "控制",
  founded: "创办",
  works_at: "任职于",
  related_to: "相关",
  headquartered_in: "总部位于",
  partner_of: "合作",
  supplier_to: "供应",
  invested_in: "投资",
  owns: "持有"
};

/** Prefer an exact raw-vocabulary label, then fall back to the typed relation. */
export function relationDisplayLabel(edge: GraphEdge): string {
  const rawRelation = edge.properties.raw_relation;
  if (typeof rawRelation === "string" && RAW_RELATION_LABELS[rawRelation]) {
    return RAW_RELATION_LABELS[rawRelation];
  }
  return TYPED_RELATION_LABELS[edge.type];
}

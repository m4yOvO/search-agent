export type NodeType = "person" | "company" | "location";

export type RelationType =
  | "controls"
  | "founded"
  | "works_at"
  | "related_to"
  | "headquartered_in"
  | "partner_of"
  | "supplier_to"
  | "invested_in"
  | "owns";

export interface GraphNode {
  id: string;
  type: NodeType;
  label: string;
  properties: Record<string, unknown>;
  evidence_ids: string[];
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: RelationType;
  label: string;
  properties: Record<string, unknown>;
  evidence_ids: string[];
}

export interface Evidence {
  id: string;
  provider: string;
  record_id: string;
  source_kind: string;
  updated_at: string;
  retrieved_at: string;
  is_demo: boolean;
  source_url: string | null;
}

export interface GraphPayload {
  graph_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  evidence: Evidence[];
  generated_at: string;
  data_version: string;
}

export interface MemoryMetadata {
  cache_hit: boolean;
  tier: string | null;
  match_type: string | null;
  status: "warm" | "hot" | "stale" | null;
  write_operation: "add" | "touch" | "promote" | "skip" | "none";
  result_id: string | null;
  reason: string | null;
}

export interface TraceMetadata {
  researcher_invoked: boolean;
  tool_calls: number;
  research_steps: number;
  replans: number;
  model_provider: string;
  model_name: string | null;
  model_calls: number;
  planner_model_calls: number;
  researcher_model_calls: number;
  visualizer_model_calls: number;
  prompt_versions: Record<string, string>;
  route_history: string[];
}

export interface ChatResponse {
  conversation_id: string;
  request_id: string;
  status: "success" | "clarification" | "failed";
  error_code:
    | "model_failure"
    | "planning_failure"
    | "research_failure"
    | "tool_failure"
    | "agent_failure"
    | null;
  answer: string;
  graph_id: string;
  graph: GraphPayload;
  memory: MemoryMetadata;
  trace: TraceMetadata;
  disclaimer: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  memory?: MemoryMetadata;
  trace?: TraceMetadata;
  createdAt: Date;
  failed?: boolean;
}

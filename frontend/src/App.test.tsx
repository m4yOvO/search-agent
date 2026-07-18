import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

vi.mock("./components/InteractiveGraph", () => ({
  InteractiveGraph: ({
    graph,
    selectedId,
    onSelect
  }: {
    graph: { nodes: { id: string }[]; edges: unknown[] };
    selectedId: string | null;
    onSelect: (id: string) => void;
  }) => (
    <div data-testid="network-stub" data-selected-id={selectedId ?? ""}>
      {graph.nodes.length} nodes / {graph.edges.length} edges
      {graph.nodes[0] ? (
        <button type="button" onClick={() => onSelect(graph.nodes[0].id)}>
          选择第一个图谱节点
        </button>
      ) : null}
    </div>
  )
}));

const FIRST_RESPONSE = {
  conversation_id: "11111111-1111-4111-8111-111111111111",
  request_id: "request:test",
  status: "success",
  error_code: null,
  answer: "原始演示数据中，马云创办了阿里巴巴集团。",
  graph_id: "graph:first",
  graph: {
    graph_id: "graph:first",
    generated_at: "2026-07-17T00:00:00Z",
    data_version: "raw-v1",
    evidence: [
      {
        id: "evidence:raw:person:P004",
        provider: "local-raw-json-mock",
        record_id: "P004",
        source_kind: "raw_person",
        updated_at: "2026-07-17T00:00:00Z",
        retrieved_at: "2026-07-17T00:00:00Z",
        is_demo: true,
        source_url: null
      },
      {
        id: "evidence:raw:company:C005",
        provider: "local-raw-json-mock",
        record_id: "C005",
        source_kind: "raw_company",
        updated_at: "2026-07-17T00:00:00Z",
        retrieved_at: "2026-07-17T00:00:00Z",
        is_demo: true,
        source_url: null
      },
      {
        id: "evidence:raw:relation:0006",
        provider: "local-raw-json-mock",
        record_id: "relations 1.json#6",
        source_kind: "raw_relation",
        updated_at: "2026-07-17T00:00:00Z",
        retrieved_at: "2026-07-17T00:00:00Z",
        is_demo: true,
        source_url: null
      }
    ],
    nodes: [
      {
        id: "person:P004",
        type: "person",
        label: "马云",
        properties: { nationality: "China" },
        evidence_ids: ["evidence:raw:person:P004"]
      },
      {
        id: "company:C005",
        type: "company",
        label: "阿里巴巴集团",
        properties: { founded_year: 1999 },
        evidence_ids: ["evidence:raw:company:C005"]
      }
    ],
    edges: [
      {
        id: "relation:raw:0006",
        source: "person:P004",
        target: "company:C005",
        type: "founded",
        label: "Founder_of",
        properties: { raw_relation: "Founder_of", source_row: 6 },
        evidence_ids: ["evidence:raw:relation:0006"]
      }
    ]
  },
  memory: {
    cache_hit: false,
    tier: null,
    match_type: null,
    status: "warm",
    write_operation: "add",
    result_id: "canonical:test",
    reason: "first_verified_result"
  },
  trace: {
    researcher_invoked: true,
    tool_calls: 2,
    research_steps: 2,
    replans: 0,
    model_provider: "openai",
    model_name: "gpt-5.4-mini",
    model_calls: 5,
    planner_model_calls: 1,
    researcher_model_calls: 3,
    visualizer_model_calls: 1,
    prompt_versions: {
      planner: "enterprise-agents-v6:planner",
      researcher: "enterprise-agents-v6:researcher",
      visualizer: "enterprise-agents-v6:visualizer"
    },
    route_history: ["planner", "researcher", "visualizer"]
  },
  disclaimer: "结果来自本地演示数据，不代表实时工商或法律结论。"
};

function jsonResponse(body: unknown = FIRST_RESPONSE): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" }
  });
}

describe("App", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("runs an example query and renders answer, cache state, and graph counts", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse());
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "马云创办了哪些公司？" }));

    expect(await screen.findByText(/马云创办了阿里巴巴集团/)).toBeInTheDocument();
    expect(screen.getByText("已写入长期记忆 · WARM")).toBeInTheDocument();
    expect(await screen.findByTestId("network-stub")).toHaveTextContent("2 nodes / 1 edges");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("/chat");
  });

  it("does not mount the lazy graph engine while the graph is empty", () => {
    render(<App />);

    expect(screen.queryByTestId("network-stub")).not.toBeInTheDocument();
    expect(screen.getByLabelText("企业关系图，当前暂无节点或关系。")).toBeInTheDocument();
  });

  it("does not submit Enter while an IME is composing", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse());
    render(<App />);
    const composer = screen.getByLabelText("继续探索");

    fireEvent.change(composer, { target: { value: "马云创办了哪些公司？" } });
    fireEvent.keyDown(composer, {
      key: "Enter",
      code: "Enter",
      keyCode: 13,
      isComposing: true
    });
    fireEvent.keyDown(composer, {
      key: "Enter",
      code: "Enter",
      keyCode: 229,
      isComposing: false
    });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(composer).toHaveValue("马云创办了哪些公司？");

    fireEvent.keyDown(composer, { key: "Enter", code: "Enter", keyCode: 13 });
    expect(await screen.findByText(/马云创办了阿里巴巴集团/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("sends the conversation ID with a follow-up", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse()).mockResolvedValueOnce(jsonResponse({
      ...FIRST_RESPONSE,
      request_id: "request:second",
      answer: "这些公司的演示所在地为：阿里巴巴集团：Hangzhou。"
    }));
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "马云创办了哪些公司？" }));
    await screen.findByText(/马云创办了阿里巴巴集团/);
    await user.type(screen.getByLabelText("继续探索"), "这些公司在哪？");
    await user.click(screen.getByRole("button", { name: "发送查询" }));

    await screen.findByText(/演示所在地/);
    const request = JSON.parse(String(fetchMock.mock.calls[1][1]?.body));
    expect(request.conversation_id).toBe(FIRST_RESPONSE.conversation_id);
    expect(request.message).toBe("这些公司在哪？");
  });

  it("shows real source and update metadata for a selected node", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse());
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "马云创办了哪些公司？" }));
    await user.click(await screen.findByRole("button", { name: "选择第一个图谱节点" }));

    expect(screen.getAllByText("local-raw-json-mock")).toHaveLength(2);
    expect(screen.getByText(/raw_person · 记录 P004/)).toBeInTheDocument();
    expect(screen.getAllByText(/数据更新/)).toHaveLength(2);
  });

  it("uses keyboard-operable text nodes to open and synchronize the inspector", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse());
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "马云创办了哪些公司？" }));
    await screen.findByText(/马云创办了阿里巴巴集团/);
    await user.click(screen.getByText("查看图谱文字列表"));
    expect(screen.getByText("马云 — 创办 → 阿里巴巴集团")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "选择企业：阿里巴巴集团" }));

    expect(screen.getByRole("heading", { name: "阿里巴巴集团" })).toBeInTheDocument();
    expect(screen.getByTestId("network-stub")).toHaveAttribute(
      "data-selected-id",
      "company:C005"
    );
  });

  it("shows a recoverable error and retries the same query", async () => {
    fetchMock
      .mockRejectedValueOnce(new TypeError("offline"))
      .mockResolvedValueOnce(jsonResponse());
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "马云创办了哪些公司？" }));
    expect(await screen.findByText(/无法连接到探索服务/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "重试这条查询" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(/马云创办了阿里巴巴集团/)).toBeInTheDocument();
  });

  it("treats a typed agent failure response as retryable", async () => {
    fetchMock
      .mockResolvedValueOnce(
        jsonResponse({
          ...FIRST_RESPONSE,
          status: "failed",
          error_code: "model_failure",
          answer: "本次查询未能从本地演示工具生成可验证结果。",
          graph: { ...FIRST_RESPONSE.graph, nodes: [], edges: [], evidence: [] }
        })
      )
      .mockResolvedValueOnce(jsonResponse());
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "马云创办了哪些公司？" }));
    expect(await screen.findByText(/未能从本地演示工具/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "重试这条查询" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(/马云创办了阿里巴巴集团/)).toBeInTheDocument();
    const retryRequest = JSON.parse(String(fetchMock.mock.calls[1][1]?.body));
    expect(retryRequest.conversation_id).toBe(FIRST_RESPONSE.conversation_id);
  });
});

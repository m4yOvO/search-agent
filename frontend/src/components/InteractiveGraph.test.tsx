import { act, fireEvent, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { GraphPayload } from "../types";
import { InteractiveGraph } from "./InteractiveGraph";

const visState = vi.hoisted(() => ({
  dataSets: [] as Array<{
    items: Map<string | number, Record<string, unknown>>;
    update: ReturnType<typeof vi.fn>;
  }>,
  networks: [] as Array<{
    handlers: Map<string, Set<(...args: unknown[]) => void>>;
    onceHandlers: Map<string, Set<(...args: unknown[]) => void>>;
    destroy: ReturnType<typeof vi.fn>;
    emit: (event: string, payload?: unknown) => void;
    fit: ReturnType<typeof vi.fn>;
    moveNode: ReturnType<typeof vi.fn>;
    off: ReturnType<typeof vi.fn>;
    setOptions: ReturnType<typeof vi.fn>;
    startSimulation: ReturnType<typeof vi.fn>;
    stopSimulation: ReturnType<typeof vi.fn>;
  }>
}));

vi.mock("vis-data", () => {
  class DataSet {
    items = new Map<string | number, Record<string, unknown>>();
    update = vi.fn((input: Record<string, unknown> | Array<Record<string, unknown>>) => {
      const records = Array.isArray(input) ? input : [input];
      for (const record of records) {
        const id = record.id as string | number;
        this.items.set(id, { ...(this.items.get(id) ?? {}), ...record });
      }
      return records.map((record) => record.id);
    });

    constructor() {
      visState.dataSets.push(this);
    }

    get length() {
      return this.items.size;
    }

    get(id: string | number) {
      return this.items.get(id) ?? null;
    }

    getIds() {
      return [...this.items.keys()];
    }

    remove(ids: Array<string | number>) {
      ids.forEach((id) => this.items.delete(id));
      return ids;
    }
  }
  return { DataSet };
});

vi.mock("vis-network", () => {
  class Network {
    handlers = new Map<string, Set<(...args: unknown[]) => void>>();
    onceHandlers = new Map<string, Set<(...args: unknown[]) => void>>();
    destroy = vi.fn();
    fit = vi.fn();
    moveNode = vi.fn();
    selectNodes = vi.fn();
    setOptions = vi.fn();
    startSimulation = vi.fn();
    stopSimulation = vi.fn();
    unselectAll = vi.fn();
    on = vi.fn((event: string, handler: (...args: unknown[]) => void) => {
      const handlers = this.handlers.get(event) ?? new Set();
      handlers.add(handler);
      this.handlers.set(event, handlers);
    });
    once = vi.fn((event: string, handler: (...args: unknown[]) => void) => {
      const handlers = this.onceHandlers.get(event) ?? new Set();
      handlers.add(handler);
      this.onceHandlers.set(event, handlers);
    });
    off = vi.fn((event: string, handler: (...args: unknown[]) => void) => {
      this.handlers.get(event)?.delete(handler);
      this.onceHandlers.get(event)?.delete(handler);
    });

    constructor() {
      visState.networks.push(this);
    }

    emit(event: string, payload?: unknown) {
      this.handlers.get(event)?.forEach((handler) => handler(payload));
      const once = [...(this.onceHandlers.get(event) ?? [])];
      this.onceHandlers.delete(event);
      once.forEach((handler) => handler(payload));
    }

    getPositions(ids: Array<string | number>) {
      return Object.fromEntries(ids.map((id, index) => [String(id), { x: index * 20, y: index * 10 }]));
    }
  }
  return { Network };
});

const graph: GraphPayload = {
  graph_id: "graph:layout",
  generated_at: "2026-07-18T00:00:00Z",
  data_version: "raw-v1",
  evidence: [],
  nodes: [
    { id: "person:P001", type: "person", label: "Elon Musk", properties: {}, evidence_ids: [] },
    { id: "company:C001", type: "company", label: "Tesla, Inc.", properties: {}, evidence_ids: [] }
  ],
  edges: [
    {
      id: "relation:1",
      source: "person:P001",
      target: "company:C001",
      type: "works_at",
      label: "CEO_of",
      properties: { raw_relation: "CEO_of" },
      evidence_ids: []
    }
  ]
};

const originalMatchMedia = Object.getOwnPropertyDescriptor(window, "matchMedia");

describe("InteractiveGraph", () => {
  beforeEach(() => {
    visState.dataSets.length = 0;
    visState.networks.length = 0;
  });

  afterEach(() => {
    vi.useRealTimers();
    if (originalMatchMedia) {
      Object.defineProperty(window, "matchMedia", originalMatchMedia);
    } else {
      Reflect.deleteProperty(window, "matchMedia");
    }
  });

  it("updates incrementally, stops physics after stabilization, and cleans up", async () => {
    vi.useFakeTimers();
    const onSelect = vi.fn();
    const view = render(
      <InteractiveGraph graph={graph} selectedId={null} onSelect={onSelect} />
    );
    const network = visState.networks[0];
    const nodes = visState.dataSets[0];
    const edges = visState.dataSets[1];

    expect(network.startSimulation).toHaveBeenCalledTimes(1);
    expect(nodes.update).toHaveBeenCalled();
    expect(edges.update).toHaveBeenCalled();
    expect(network.setOptions).toHaveBeenLastCalledWith(
      expect.objectContaining({ physics: expect.objectContaining({ enabled: true }) })
    );

    network.emit("stabilized");
    expect(network.stopSimulation).toHaveBeenCalled();
    expect(network.setOptions).toHaveBeenLastCalledWith(
      expect.objectContaining({ physics: { enabled: false, stabilization: false } })
    );

    const nodeUpdates = nodes.update.mock.calls.length;
    const edgeUpdates = edges.update.mock.calls.length;
    view.rerender(
      <InteractiveGraph
        graph={{ ...graph, nodes: [...graph.nodes], edges: [...graph.edges] }}
        selectedId={null}
        onSelect={onSelect}
      />
    );
    expect(nodes.update).toHaveBeenCalledTimes(nodeUpdates);
    expect(edges.update).toHaveBeenCalledTimes(edgeUpdates);

    fireEvent.click(view.getByRole("button", { name: "重新布局" }));
    expect(network.moveNode).toHaveBeenCalledTimes(graph.nodes.length);
    expect(network.startSimulation).toHaveBeenCalledTimes(2);
    act(() => vi.advanceTimersByTime(1_200));
    expect(network.setOptions).toHaveBeenLastCalledWith(
      expect.objectContaining({ physics: { enabled: false, stabilization: false } })
    );

    view.unmount();
    expect(network.destroy).toHaveBeenCalledTimes(1);
    expect(network.off).toHaveBeenCalledWith("selectNode", expect.any(Function));
    expect(network.off).toHaveBeenCalledWith("deselectNode", expect.any(Function));
  });

  it("keeps reduced-motion layout static, including manual relayout", () => {
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn(() => ({
        matches: true,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn()
      }))
    });

    const view = render(
      <InteractiveGraph graph={graph} selectedId={null} onSelect={vi.fn()} />
    );
    const network = visState.networks[0];

    expect(network.startSimulation).not.toHaveBeenCalled();
    fireEvent.click(view.getByRole("button", { name: "重新布局" }));
    expect(network.moveNode).toHaveBeenCalledTimes(graph.nodes.length);
    expect(network.startSimulation).not.toHaveBeenCalled();
    expect(network.fit).toHaveBeenLastCalledWith({ animation: false });

    view.unmount();
  });
});

import { describe, expect, it } from "vitest";

import {
  fitOptionsForMotion,
  graphToVis,
  networkOptionsForMotion,
  toVisEdge,
  toVisNode
} from "./graphAdapter";
import { relationDisplayLabel } from "./relationPresentation";
import type { GraphEdge } from "./types";
import type { GraphPayload } from "./types";

const graph: GraphPayload = {
  graph_id: "graph:test",
  generated_at: "2026-07-17T00:00:00Z",
  data_version: "demo-v1",
  evidence: [
    {
      id: "evidence:person:P001",
      provider: "local-demo-fixture",
      record_id: "P001",
      source_kind: "raw_person",
      updated_at: "2026-07-17T00:00:00Z",
      retrieved_at: "2026-07-17T00:00:00Z",
      is_demo: true,
      source_url: null
    },
    {
      id: "evidence:company:C001",
      provider: "local-demo-fixture",
      record_id: "C001",
      source_kind: "raw_company",
      updated_at: "2026-07-17T00:00:00Z",
      retrieved_at: "2026-07-17T00:00:00Z",
      is_demo: true,
      source_url: null
    },
    {
      id: "evidence:location:austin",
      provider: "local-demo-fixture",
      record_id: "Austin",
      source_kind: "normalized_location",
      updated_at: "2026-07-17T00:00:00Z",
      retrieved_at: "2026-07-17T00:00:00Z",
      is_demo: true,
      source_url: null
    },
    {
      id: "evidence:relation:controls",
      provider: "local-demo-fixture",
      record_id: "controls",
      source_kind: "demo_relation",
      updated_at: "2026-07-17T00:00:00Z",
      retrieved_at: "2026-07-17T00:00:00Z",
      is_demo: true,
      source_url: null
    }
  ],
  nodes: [
    {
      id: "person:P001",
      type: "person",
      label: "Elon Musk",
      properties: {},
      evidence_ids: ["evidence:person:P001"]
    },
    {
      id: "company:C001",
      type: "company",
      label: "Tesla, Inc.",
      properties: {},
      evidence_ids: ["evidence:company:C001"]
    },
    {
      id: "location:austin",
      type: "location",
      label: "Austin",
      properties: {},
      evidence_ids: ["evidence:location:austin"]
    }
  ],
  edges: [
    {
      id: "relation:controls",
      source: "person:P001",
      target: "company:C001",
      type: "controls",
      label: "controls",
      properties: {},
      evidence_ids: ["evidence:relation:controls"]
    }
  ]
};

describe("graph adapter", () => {
  it("maps backend source/target to vis-network from/to", () => {
    const edge = toVisEdge(graph.edges[0]);

    expect(edge.from).toBe("person:P001");
    expect(edge.to).toBe("company:C001");
    expect(edge).not.toHaveProperty("source");
    expect(edge).not.toHaveProperty("target");
    expect(edge.smooth).toMatchObject({ type: "cubicBezier" });
  });

  it("localizes controlled raw relations before using the typed fallback", () => {
    const edge: GraphEdge = {
      ...graph.edges[0],
      type: "works_at",
      label: "Former_CEO_of",
      properties: { raw_relation: "Former_CEO_of", source_row: 17 }
    };

    expect(relationDisplayLabel(edge)).toBe("曾任 CEO");
    expect(toVisEdge(edge).label).toBe("曾任 CEO");
    expect(edge.label).toBe("Former_CEO_of");
    expect(edge.properties).toEqual({ raw_relation: "Former_CEO_of", source_row: 17 });
  });

  it.each([
    ["Founder_of", "创办"],
    ["Co-founder_of", "联合创办"],
    ["CEO_of", "担任 CEO"],
    ["Former_Chairman_of", "曾任董事长"],
    ["Headquartered_in", "总部位于"]
  ])("renders %s as %s", (rawRelation, expected) => {
    expect(relationDisplayLabel({
      ...graph.edges[0],
      properties: { raw_relation: rawRelation }
    })).toBe(expected);
  });

  it("falls back to the typed relation for an unknown or absent raw relation", () => {
    expect(relationDisplayLabel({
      ...graph.edges[0],
      type: "headquartered_in",
      properties: { raw_relation: "Future_Source_Vocabulary" }
    })).toBe("总部位于");
    expect(relationDisplayLabel({
      ...graph.edges[0],
      type: "owns",
      properties: {}
    })).toBe("持有");
  });

  it("uses distinct shapes in addition to color", () => {
    expect(toVisNode(graph.nodes[0]).shape).toBe("dot");
    expect(toVisNode(graph.nodes[1]).shape).toBe("box");
    expect(toVisNode(graph.nodes[2]).shape).toBe("diamond");
  });

  it("keeps stable backend IDs for incremental updates", () => {
    const visual = graphToVis(graph);
    expect(visual.nodes.map((node) => node.id)).toEqual([
      "person:P001",
      "company:C001",
      "location:austin"
    ]);
    expect(visual.edges[0].id).toBe("relation:controls");
  });

  it("turns off physics and camera animation for reduced motion", () => {
    expect(networkOptionsForMotion(true).physics).toEqual({
      enabled: false,
      stabilization: false
    });
    expect(fitOptionsForMotion(true)).toEqual({ animation: false });
    expect(fitOptionsForMotion(false, 320)).toEqual({
      animation: { duration: 320, easingFunction: "easeInOutQuad" }
    });
  });
});

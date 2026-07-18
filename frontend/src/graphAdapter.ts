import type { Edge, FitOptions, Node, Options } from "vis-network";

import type { GraphEdge, GraphNode, GraphPayload, NodeType } from "./types";
import { relationDisplayLabel } from "./relationPresentation";

const NODE_VISUALS: Record<
  NodeType,
  Pick<Node, "shape" | "color" | "font" | "borderWidth" | "size" | "margin">
> = {
  person: {
    shape: "dot",
    color: {
      background: "#19324a",
      border: "#102335",
      highlight: { background: "#234968", border: "#0b776f" },
      hover: { background: "#234968", border: "#0b776f" }
    },
    font: { color: "#102335", face: "Inter, PingFang SC, sans-serif", size: 15 },
    borderWidth: 2,
    size: 23,
    margin: { top: 12, right: 12, bottom: 12, left: 12 }
  },
  company: {
    shape: "box",
    color: {
      background: "#f8f4e9",
      border: "#0b776f",
      highlight: { background: "#e4f1ed", border: "#075f59" },
      hover: { background: "#eef6f3", border: "#075f59" }
    },
    font: { color: "#102335", face: "Inter, PingFang SC, sans-serif", size: 15 },
    borderWidth: 2,
    size: 24,
    margin: { top: 11, right: 15, bottom: 11, left: 15 }
  },
  location: {
    shape: "diamond",
    color: {
      background: "#d99a28",
      border: "#8d621b",
      highlight: { background: "#efb54c", border: "#704b10" },
      hover: { background: "#efb54c", border: "#704b10" }
    },
    font: { color: "#102335", face: "Inter, PingFang SC, sans-serif", size: 14 },
    borderWidth: 2,
    size: 20,
    margin: { top: 10, right: 10, bottom: 10, left: 10 }
  }
};

export function toVisNode(node: GraphNode): Node {
  return {
    id: node.id,
    label: node.label,
    ...NODE_VISUALS[node.type]
  };
}

export function toVisEdge(edge: GraphEdge): Edge {
  return {
    id: edge.id,
    from: edge.source,
    to: edge.target,
    label: relationDisplayLabel(edge),
    arrows: { to: { enabled: true, scaleFactor: 0.62 } },
    color: {
      color: edge.type === "headquartered_in" ? "#a66f16" : "#55706d",
      highlight: "#075f59",
      hover: "#075f59",
      opacity: 0.92
    },
    dashes: edge.type === "related_to",
    width: 1.7,
    selectionWidth: 2.4,
    font: {
      color: "#52615f",
      face: "ui-monospace, SFMono-Regular, Menlo, monospace",
      size: 10,
      align: "middle",
      background: "#f6f2e8",
      strokeWidth: 0
    },
    // Avoid vis-network's continuously recalculated dynamic support nodes.
    smooth: { enabled: true, type: "cubicBezier", roundness: 0.22 }
  };
}

export function graphToVis(graph: GraphPayload): { nodes: Node[]; edges: Edge[] } {
  return {
    nodes: graph.nodes.map(toVisNode),
    edges: graph.edges.map(toVisEdge)
  };
}

const BASE_NETWORK_OPTIONS: Options = {
  autoResize: true,
  interaction: {
    hover: true,
    keyboard: { enabled: true, bindToWindow: false },
    navigationButtons: false,
    tooltipDelay: 300
  },
  layout: { improvedLayout: true, randomSeed: 17 },
  physics: {
    enabled: true,
    stabilization: { enabled: true, iterations: 140, updateInterval: 25 },
    barnesHut: {
      gravitationalConstant: -3400,
      centralGravity: 0.12,
      springLength: 160,
      springConstant: 0.035,
      damping: 0.25,
      avoidOverlap: 0.6
    }
  },
  nodes: { chosen: true },
  edges: { chosen: true }
};

/**
 * Build options instead of mutating a shared vis-network object. Reduced-motion
 * users get a static layout: vis does not run or stabilize a physics simulation.
 */
export function networkOptionsForMotion(reducedMotion: boolean): Options {
  if (!reducedMotion) return BASE_NETWORK_OPTIONS;

  return {
    ...BASE_NETWORK_OPTIONS,
    physics: {
      enabled: false,
      stabilization: false
    }
  };
}

/** Keep programmatic camera movement subject to the same OS motion preference. */
export function fitOptionsForMotion(
  reducedMotion: boolean,
  duration = 360
): FitOptions {
  return reducedMotion
    ? { animation: false }
    : { animation: { duration, easingFunction: "easeInOutQuad" } };
}

export const NETWORK_OPTIONS: Options = networkOptionsForMotion(false);

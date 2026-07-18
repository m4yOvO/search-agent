import { useEffect, useRef } from "react";
import { DataSet } from "vis-data";
import { Network, type Edge, type IdType, type Node, type Position } from "vis-network";
import "vis-network/styles/vis-network.css";

import {
  fitOptionsForMotion,
  graphToVis,
  networkOptionsForMotion
} from "../graphAdapter";
import type { GraphPayload } from "../types";

interface InteractiveGraphProps {
  graph: GraphPayload;
  selectedId: string | null;
  onSelect: (nodeId: string | null) => void;
}

interface SyncResult {
  added: IdType[];
  updated: IdType[];
  removed: IdType[];
}

const LAYOUT_TIMEOUT_MS = 1_200;

/**
 * Apply only actual changes. Besides avoiding needless vis redraws, keeping an
 * unchanged node out of DataSet.update lets vis retain its canvas position.
 */
function syncDataSet<T extends { id?: IdType }>(
  dataSet: DataSet<T, "id">,
  next: T[],
  fingerprints: Map<IdType, string>
): SyncResult {
  const nextIds = new Set(next.map((item) => item.id).filter((id): id is IdType => id != null));
  const removed = dataSet.getIds().filter((id) => !nextIds.has(id));
  const added: IdType[] = [];
  const updated: IdType[] = [];
  const changed: T[] = [];

  for (const item of next) {
    if (item.id == null) continue;
    const fingerprint = JSON.stringify(item);
    const previous = fingerprints.get(item.id);
    if (previous == null) {
      added.push(item.id);
      changed.push(item);
    } else if (previous !== fingerprint) {
      updated.push(item.id);
      changed.push(item);
    }
    fingerprints.set(item.id, fingerprint);
  }

  if (removed.length) {
    dataSet.remove(removed);
    removed.forEach((id) => fingerprints.delete(id));
  }
  if (changed.length) {
    dataSet.update(changed as Parameters<typeof dataSet.update>[0]);
  }
  return { added, updated, removed };
}

function prefersReducedMotion(): boolean {
  return typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function InteractiveGraph({ graph, selectedId, onSelect }: InteractiveGraphProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const networkRef = useRef<Network | null>(null);
  const nodesRef = useRef(new DataSet<Node, "id">());
  const edgesRef = useRef(new DataSet<Edge, "id">());
  const nodeFingerprintsRef = useRef(new Map<IdType, string>());
  const edgeFingerprintsRef = useRef(new Map<IdType, string>());
  const resizeFitTimeoutRef = useRef<number | null>(null);
  const layoutTimeoutRef = useRef<number | null>(null);
  const stabilizedHandlerRef = useRef<(() => void) | null>(null);
  const fixedNodeIdsRef = useRef<IdType[]>([]);
  const layoutGenerationRef = useRef(0);
  const relayoutSequenceRef = useRef(0);
  const reducedMotionRef = useRef(prefersReducedMotion());
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  function releaseFixedNodes() {
    const ids = fixedNodeIdsRef.current.filter((id) => nodesRef.current.get(id) != null);
    fixedNodeIdsRef.current = [];
    if (ids.length) {
      nodesRef.current.update(
        ids.map((id) => ({ id, fixed: false })) as Parameters<typeof nodesRef.current.update>[0]
      );
    }
  }

  function cancelActiveLayout(network: Network) {
    layoutGenerationRef.current += 1;
    if (layoutTimeoutRef.current != null) {
      window.clearTimeout(layoutTimeoutRef.current);
      layoutTimeoutRef.current = null;
    }
    if (stabilizedHandlerRef.current) {
      network.off("stabilized", stabilizedHandlerRef.current);
      stabilizedHandlerRef.current = null;
    }
    // stopSimulation emits `stabilized`, so detach and invalidate first.
    network.stopSimulation();
    network.setOptions(networkOptionsForMotion(true));
    releaseFixedNodes();
  }

  function finishLayout(network: Network, generation: number, fit = true) {
    if (networkRef.current !== network || layoutGenerationRef.current !== generation) return;
    cancelActiveLayout(network);
    if (fit && nodesRef.current.length > 0) {
      network.fit(fitOptionsForMotion(reducedMotionRef.current));
    }
  }

  function startShortLayout(network: Network, preservedPositions: Record<string, Position> = {}) {
    cancelActiveLayout(network);

    const fixedItems = Object.entries(preservedPositions)
      .filter(([id]) => nodesRef.current.get(id) != null)
      .map(([id, position]) => ({
        id,
        x: position.x,
        y: position.y,
        fixed: { x: true, y: true }
      }));
    fixedNodeIdsRef.current = fixedItems.map((item) => item.id);
    if (fixedItems.length) {
      nodesRef.current.update(
        fixedItems as Parameters<typeof nodesRef.current.update>[0]
      );
    }

    if (reducedMotionRef.current) {
      releaseFixedNodes();
      network.fit(fitOptionsForMotion(true));
      return;
    }

    const generation = ++layoutGenerationRef.current;
    const onStabilized = () => finishLayout(network, generation);
    stabilizedHandlerRef.current = onStabilized;
    network.once("stabilized", onStabilized);
    network.setOptions(networkOptionsForMotion(false));
    network.startSimulation();
    layoutTimeoutRef.current = window.setTimeout(
      () => finishLayout(network, generation),
      LAYOUT_TIMEOUT_MS
    );
  }

  useEffect(() => {
    if (!containerRef.current) return;

    // The network starts static. Physics is enabled only by startShortLayout.
    const network = new Network(
      containerRef.current,
      { nodes: nodesRef.current, edges: edgesRef.current },
      networkOptionsForMotion(true)
    );
    networkRef.current = network;
    const selectNode = ({ nodes }: { nodes: IdType[] }) => {
      onSelectRef.current(nodes[0] == null ? null : String(nodes[0]));
    };
    const deselectNode = () => onSelectRef.current(null);
    network.on("selectNode", selectNode);
    network.on("deselectNode", deselectNode);

    return () => {
      if (resizeFitTimeoutRef.current != null) {
        window.clearTimeout(resizeFitTimeoutRef.current);
        resizeFitTimeoutRef.current = null;
      }
      cancelActiveLayout(network);
      network.off("selectNode", selectNode);
      network.off("deselectNode", deselectNode);
      network.destroy();
      if (networkRef.current === network) networkRef.current = null;
    };
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") return;

    let previousWidth = container.clientWidth;
    let previousHeight = container.clientHeight;
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      const changed = Math.abs(width - previousWidth) > 1 || Math.abs(height - previousHeight) > 1;
      previousWidth = width;
      previousHeight = height;
      if (!changed || nodesRef.current.length === 0) return;

      if (resizeFitTimeoutRef.current != null) {
        window.clearTimeout(resizeFitTimeoutRef.current);
      }
      resizeFitTimeoutRef.current = window.setTimeout(() => {
        networkRef.current?.fit(fitOptionsForMotion(reducedMotionRef.current, 240));
        resizeFitTimeoutRef.current = null;
      }, 180);
    });

    observer.observe(container);
    return () => {
      observer.disconnect();
      if (resizeFitTimeoutRef.current != null) {
        window.clearTimeout(resizeFitTimeoutRef.current);
        resizeFitTimeoutRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const applyPreference = (matches: boolean) => {
      reducedMotionRef.current = matches;
      const network = networkRef.current;
      if (!network) return;
      if (matches) cancelActiveLayout(network);
      else network.setOptions(networkOptionsForMotion(true));
    };
    const onChange = (event: MediaQueryListEvent) => applyPreference(event.matches);

    applyPreference(query.matches);
    query.addEventListener?.("change", onChange);
    return () => query.removeEventListener?.("change", onChange);
  }, []);

  useEffect(() => {
    const previousNodeIds = nodesRef.current.getIds();
    const network = networkRef.current;
    const preservedPositions = network && previousNodeIds.length
      ? network.getPositions(previousNodeIds)
      : {};
    const visual = graphToVis(graph);
    const nodeChanges = syncDataSet(
      nodesRef.current,
      visual.nodes,
      nodeFingerprintsRef.current
    );
    syncDataSet(edgesRef.current, visual.edges, edgeFingerprintsRef.current);

    if (!network || visual.nodes.length === 0) {
      if (network && visual.nodes.length === 0) cancelActiveLayout(network);
      return;
    }

    if (nodeChanges.added.length > 0) {
      // New nodes may settle around the old graph; prior nodes are pinned for
      // this short pass and released after physics has been disabled again.
      startShortLayout(network, preservedPositions);
    } else if (nodeChanges.removed.length > 0) {
      network.fit(fitOptionsForMotion(reducedMotionRef.current));
    }
  }, [graph]);

  useEffect(() => {
    if (!networkRef.current) return;
    if (selectedId && nodesRef.current.get(selectedId)) {
      networkRef.current.selectNodes([selectedId], false);
    } else {
      networkRef.current.unselectAll();
    }
  }, [selectedId, graph.nodes]);

  function fitView() {
    networkRef.current?.fit(fitOptionsForMotion(reducedMotionRef.current, 320));
  }

  function relayout() {
    const network = networkRef.current;
    const ids = nodesRef.current.getIds().sort((left, right) => String(left).localeCompare(String(right)));
    if (!network || ids.length === 0) return;

    relayoutSequenceRef.current += 1;
    const radius = Math.max(110, Math.min(260, ids.length * 34));
    const rotation = relayoutSequenceRef.current * 0.37;
    ids.forEach((id, index) => {
      const angle = rotation + (Math.PI * 2 * index) / ids.length;
      network.moveNode(id, Math.cos(angle) * radius, Math.sin(angle) * radius);
    });

    // Reduced-motion users get the new deterministic layout immediately.
    startShortLayout(network);
  }

  return (
    <div className="graph-canvas-shell">
      <div className="graph-controls">
        <button className="graph-fit-button" type="button" onClick={fitView}>
          <span aria-hidden="true">⌗</span>
          适合视图
        </button>
        <button className="graph-fit-button" type="button" onClick={relayout}>
          <span aria-hidden="true">↻</span>
          重新布局
        </button>
      </div>
      <div
        ref={containerRef}
        className="graph-canvas"
        role="img"
        tabIndex={0}
        aria-label={`企业关系图，共 ${graph.nodes.length} 个节点、${graph.edges.length} 条关系。可拖拽、缩放并选择节点。`}
      />
      {graph.nodes.length === 0 ? (
        <div className="graph-empty" aria-hidden="true">
          <span className="graph-empty-mark">◎</span>
          <p>关系图等待第一条查询</p>
          <small>回答中的人物、企业与地点会在这里连接起来</small>
        </div>
      ) : null}
    </div>
  );
}

"""Bounded approximate token radix trees, scoped by tenant/model and replica.

Compressed edges permit matches ending *inside* an edge, not just at stored
checkpoints. Only successful completions populate the index. This tracks request
history, not actual KV residency; no engine events are consumed.
"""

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

TreeKey = tuple[str, str, int]  # Trusted namespace, worker ID, generation.


@dataclass(eq=False, slots=True)
class Node:
    edge: tuple[int, ...] = ()
    parent: "Node | None" = None
    children: dict[int, "Node"] = field(default_factory=dict)


class ApproximatePrefixIndex:
    def __init__(
        self,
        capacity=4096,
        ttl=30,
        max_tokens=65_536,
        max_nodes=8192,
        max_prefix_tokens=4096,
        clock=time.monotonic,
    ):
        if min(capacity, max_tokens, max_nodes, max_prefix_tokens) < 1:
            raise ValueError("Prefix index limits must be positive")
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("Prefix index TTL must be positive")
        self.capacity, self.ttl, self.clock = capacity, ttl, clock
        self.max_tokens, self.max_nodes = max_tokens, max_nodes
        self.max_prefix_tokens = max_prefix_tokens
        self.roots: dict[TreeKey, Node] = {}
        # Recency means last successful completion, not speculative dispatch or
        # read. A terminal can also have children (a shorter completed prompt).
        self.records: OrderedDict[Node, tuple[TreeKey, float]] = OrderedDict()
        self.nodes = 0
        self.tokens = 0

    def _remove(self, node):
        key, _ = self.records.pop(node)
        root = self.roots[key]
        while node is not root and node not in self.records:
            parent = node.parent
            if not node.children:
                del parent.children[node.edge[0]]
                self.nodes -= 1
                self.tokens -= len(node.edge)
                node.parent = None
                node = parent
            elif len(node.children) == 1:
                # Preserve the child's identity: its terminal may be recorded.
                child = next(iter(node.children.values()))
                child.edge = node.edge + child.edge
                child.parent = parent
                parent.children[node.edge[0]] = child
                node.children.clear()
                node.parent = None
                self.nodes -= 1  # Merging preserves the total edge-token count.
                node = parent
            else:
                break
        if not root.children:
            del self.roots[key]
            self.nodes -= 1

    def prune(self):
        now = self.clock()
        for node, (_, expiry) in list(self.records.items()):
            if expiry <= now:
                self._remove(node)

    def invalidate(self, worker):
        for node, (key, _) in list(self.records.items()):
            if key[1] == worker:
                self._remove(node)

    def remember(self, namespace, tokens: tuple[int, ...], worker, generation):
        if not namespace or not tokens:
            return
        self.prune()
        tokens = tokens[: self.max_prefix_tokens]
        key = namespace, worker, generation
        root = self.roots.get(key)
        if root is None:
            root = self.roots[key] = Node()
            self.nodes += 1
        node, offset = root, 0
        while offset < len(tokens):
            child = node.children.get(tokens[offset])
            if child is None:
                child = Node(tokens[offset:], node)
                node.children[child.edge[0]] = child
                self.nodes += 1
                self.tokens += len(child.edge)
                node = child
                break
            common = 0
            while (
                common < len(child.edge)
                and offset + common < len(tokens)
                and child.edge[common] == tokens[offset + common]
            ):
                common += 1
            if common < len(child.edge):
                middle = Node(child.edge[:common], node)
                node.children[middle.edge[0]] = middle
                child.edge = child.edge[common:]
                child.parent = middle
                middle.children[child.edge[0]] = child
                self.nodes += 1
                node = middle
            else:
                node = child
            offset += common
        self.records[node] = key, self.clock() + self.ttl
        self.records.move_to_end(node)
        while (
            len(self.records) > self.capacity
            or self.tokens > self.max_tokens
            or self.nodes > self.max_nodes
        ):
            self._remove(next(iter(self.records)))

    def match(self, namespace, tokens: tuple[int, ...], worker, generation) -> int:
        """Longest common token prefix, including a mismatch inside an edge."""
        self.prune()
        node = self.roots.get((namespace, worker, generation))
        if node is None:
            return 0
        limit = min(len(tokens), self.max_prefix_tokens)
        offset = 0
        while offset < limit:
            child = node.children.get(tokens[offset])
            if child is None:
                break
            common = 0
            while (
                common < len(child.edge)
                and offset + common < limit
                and child.edge[common] == tokens[offset + common]
            ):
                common += 1
            offset += common
            if common < len(child.edge):
                break
            node = child
        return offset

    def snapshot(self):
        self.prune()
        return {
            "kind": "approximate-token-radix-tree",
            "entries": len(self.records),
            "trees": len(self.roots),
            "nodes": self.nodes,
            "edge_tokens": self.tokens,
            "capacity": self.capacity,
            "max_nodes": self.max_nodes,
            "max_tokens": self.max_tokens,
            "max_prefix_tokens": self.max_prefix_tokens,
            "ttl_seconds": self.ttl,
            "recency": "successful-completion",
        }

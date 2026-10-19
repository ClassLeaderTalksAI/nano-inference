import random
from collections import OrderedDict

import pytest

from nano_inference.prefix_index import ApproximatePrefixIndex


def consistent(index):
    """Check counters and compressed structure independently of lookup results."""
    seen, terminals, tokens = set(), set(index.records), 0
    for key, root in index.roots.items():
        assert root.parent is None and root.edge == () and root.children
        stack = [root]
        while stack:
            node = stack.pop()
            assert node not in seen
            seen.add(node)
            tokens += len(node.edge)
            if node in terminals:
                assert index.records[node][0] == key
            elif node is not root:
                assert len(node.children) >= 2  # No removable empty/unary nodes.
            for token, child in node.children.items():
                assert child.parent is node and token == child.edge[0]
                stack.append(child)
    assert terminals <= seen
    assert len(seen) == index.nodes and tokens == index.tokens
    assert len(index.records) <= index.capacity
    assert index.nodes <= index.max_nodes and index.tokens <= index.max_tokens


def test_arbitrary_lpm_inside_compressed_edges_and_prefix_insertion():
    index = ApproximatePrefixIndex()
    index.remember("tenant-a:model", tuple(range(200)), "worker-1", 1)
    query = tuple(range(93)) + (999, 1000)
    assert index.match("tenant-a:model", query, "worker-1", 1) == 93
    assert index.match("tenant-a:model", tuple(range(50)), "worker-1", 1) == 50
    assert index.nodes == 2  # Root + one compressed 200-token edge.
    index.remember("tenant-a:model", query, "worker-1", 1)
    index.remember("tenant-a:model", tuple(range(40)), "worker-1", 1)
    assert index.match("tenant-a:model", query, "worker-1", 1) == 95
    assert index.match("tenant-a:model", tuple(range(201)), "worker-1", 1) == 200
    consistent(index)


def test_scopes_replicas_generations_and_empty_inputs_do_not_alias():
    index = ApproximatePrefixIndex()
    index.remember("tenant-a:model", (1, 2, 3), "worker-1", 1)
    index.remember("tenant-a:model", (1, 2, 4), "worker-2", 1)
    assert index.match("tenant-a:model", (1, 2, 3), "worker-1", 1) == 3
    assert index.match("tenant-a:model", (1, 2, 3), "worker-2", 1) == 2
    assert index.match("tenant-b:model", (1, 2, 3), "worker-1", 1) == 0
    assert index.match("tenant-a:other-model", (1, 2, 3), "worker-1", 1) == 0
    assert index.match("tenant-a:model", (1, 2, 3), "worker-1", 2) == 0
    index.remember("", (1,), "worker-1", 1)
    index.remember("tenant-a:model", (), "worker-1", 1)
    assert len(index.records) == 2
    consistent(index)


def test_expiring_short_terminal_recompresses_without_losing_longer_live_prompt():
    now = [0.0]
    index = ApproximatePrefixIndex(ttl=3, clock=lambda: now[0])
    index.remember("a", (1, 2), "w", 1)
    now[0] = 1
    index.remember("a", (1, 2, 3, 4), "w", 1)
    now[0] = 3
    assert index.match("a", (1, 2, 3, 5), "w", 1) == 3
    assert len(index.records) == 1 and index.nodes == 2
    consistent(index)
    now[0] = 4
    assert index.match("a", (1, 2, 3), "w", 1) == 0
    assert not index.roots and index.nodes == index.tokens == 0


def test_expired_long_branch_cannot_leave_a_false_prefix_hint():
    now = [0.0]
    index = ApproximatePrefixIndex(ttl=3, clock=lambda: now[0])
    index.remember("a", (1, 2, 3, 4), "w", 1)
    now[0] = 1
    index.remember("a", (1, 2, 9), "w", 1)
    now[0] = 3
    assert index.match("a", (1, 2, 3, 4), "w", 1) == 2
    consistent(index)
    index.invalidate("w")
    assert index.snapshot()["entries"] == 0
    assert index.nodes == index.tokens == 0


@pytest.mark.parametrize(
    "limits",
    [
        {"capacity": 2},
        {"max_tokens": 6},
        {"max_nodes": 3},
        {"max_prefix_tokens": 2},
    ],
)
def test_memory_budgets_prune_paths_not_just_accounting(limits):
    index = ApproximatePrefixIndex(**limits)
    for i in range(20):
        index.remember("a", (i, i + 1, i + 2), "w", 1)
        consistent(index)
    if "capacity" in limits:
        assert index.match("a", (0, 1, 2), "w", 1) == 0
    if "max_prefix_tokens" in limits:
        assert index.match("a", (19, 20, 21), "w", 1) == 2


def test_recency_and_ttl_refresh_on_success_only_not_reads():
    now = [0.0]
    index = ApproximatePrefixIndex(capacity=2, ttl=3, clock=lambda: now[0])
    index.remember("a", (1, 2), "w", 1)
    index.remember("a", (3, 4), "w", 1)
    now[0] = 1
    index.remember("a", (1, 2), "w", 1)  # Successful reuse refreshes this path.
    index.remember("a", (5, 6), "w", 1)
    assert index.match("a", (3, 4), "w", 1) == 0
    now[0] = 3.9
    assert index.match("a", (1, 2), "w", 1) == 2
    now[0] = 4
    assert index.match("a", (1, 2), "w", 1) == 0


def test_randomized_operations_match_independent_full_history_oracle():
    rng, now = random.Random(20261019), [0.0]
    index = ApproximatePrefixIndex(capacity=8, ttl=3, clock=lambda: now[0])
    history = OrderedDict()
    for _ in range(500):
        now[0] += rng.choice([0, 0.25, 0.5])
        history = OrderedDict((k, v) for k, v in history.items() if v > now[0])
        if rng.random() < 0.1:
            worker = rng.choice(["w1", "w2"])
            index.invalidate(worker)
            history = OrderedDict((k, v) for k, v in history.items() if k[1] != worker)
        else:
            namespace, worker, generation = (
                rng.choice(["a", "b"]),
                rng.choice(["w1", "w2"]),
                rng.choice([1, 2]),
            )
            path = tuple(rng.randrange(4) for _ in range(rng.randrange(1, 12)))
            index.remember(namespace, path, worker, generation)
            key = namespace, worker, generation, path
            history[key] = now[0] + 3
            history.move_to_end(key)
            while len(history) > 8:
                history.popitem(last=False)
        index.prune()
        consistent(index)
        assert len(index.records) == len(history)
        for _ in range(4):
            ns, worker, gen = rng.choice(["a", "b"]), rng.choice(["w1", "w2"]), rng.choice([1, 2])
            query = tuple(rng.randrange(4) for _ in range(10))
            expected = 0
            for hns, hw, hg, path in history:
                if (hns, hw, hg) != (ns, worker, gen):
                    continue
                common = 0
                for a, b in zip(path, query, strict=False):
                    if a != b:
                        break
                    common += 1
                expected = max(expected, common)
            assert index.match(ns, query, worker, gen) == expected

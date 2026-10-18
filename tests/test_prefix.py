import asyncio
import threading

import pytest

from nano_inference.prefix import AffinityTable, PrefixEncoder, PrefixPreparer
from nano_inference.protocol import Message
from nano_inference.routing import AdmissionError, WorkerPool

from .conftest import eventually


def messages(text):
    return [Message(role="system", content=text)]


def test_identity_is_stable_and_distinguishes_revision_and_prefix():
    encoder = PrefixEncoder("nano-model", "r1", tokens=64)
    first = encoder.key(messages("prefix-A " + "x" * 100))
    assert first == encoder.key(messages("prefix-A " + "x" * 100 + "different suffix"))
    assert first != encoder.key(messages("prefix-B " + "x" * 100))
    assert first != PrefixEncoder("nano-model", "r2", tokens=64).key(
        messages("prefix-A " + "x" * 100)
    )
    assert encoder.key(messages("tiny")) is None
    with pytest.raises(ValueError, match="Synthetic"):
        PrefixEncoder("real-model", "r1")


def test_identity_uses_token_ids_and_fingerprints_not_raw_text():
    class Tokenizer:
        is_fast = True
        chat_template = "template-a"

        @property
        def backend_tokenizer(self):
            return self

        def to_str(self):
            return "tokenizer-a"

        def apply_chat_template(self, data, tokenize, add_generation_prompt):
            assert tokenize and add_generation_prompt
            # Two spellings normalize to the same tokenizer output.
            return [1, 2, 3, len(data[0]["content"])]

    tokenizer = Tokenizer()
    a = PrefixEncoder("real", "sha256:weights", 3, tokenizer)
    assert a.key(messages("spelling one")) == a.key(messages("other spelling"))
    tokenizer.chat_template = "template-b"
    b = PrefixEncoder("real", "sha256:weights", 3, tokenizer)
    assert a.key(messages("same")) != b.key(messages("same"))


def test_table_ttl_lru_and_stable_home():
    now = [0.0]
    table = AffinityTable(capacity=2, ttl=10, clock=lambda: now[0])
    table.remember("a", "worker-1", 1)
    table.remember("b", "worker-2", 1)
    assert table.get("a").worker == "worker-1"
    table.remember("a", "worker-2", 1)  # A fallback completion does not change its home.
    assert table.get("a").worker == "worker-1"
    table.remember("c", "worker-2", 1)
    assert table.get("b") is None  # LRU eviction.
    now[0] = 9
    assert table.get("a") is not None
    now[0] = 10
    assert table.get("a") is None  # Reads do not keep hints alive forever.
    table.prune()
    assert not table.entries


async def prefix_pool(**kwargs):
    pool = WorkerPool(["http://a", "http://b"], policy="prefix_affinity", **kwargs)
    for worker in pool.workers:
        await pool.set_health(worker, True)
    return pool


async def test_success_warms_affinity_and_failure_does_not():
    pool = await prefix_pool()
    cold = await pool.acquire("prefix-a")
    assert cold.decision == "affinity_miss"
    await pool.release(cold, successful=True)
    warm = await pool.acquire("prefix-a")
    assert warm.worker is cold.worker
    assert warm.decision == "affinity_hit"
    await pool.release(warm)
    failed = await pool.acquire("prefix-b")
    await pool.release(failed)
    assert pool.affinity.get("prefix-b") is None


async def test_affinity_load_override_and_capacity_are_respected():
    pool = await prefix_pool(capacity=3, affinity_max_skew=1)
    seed = await pool.acquire("hot")
    await pool.release(seed, successful=True)
    one = await pool.acquire("hot")
    two = await pool.acquire("hot")
    three = await pool.acquire("hot")
    assert one.worker is two.worker is seed.worker
    assert three.worker is not seed.worker
    assert three.decision == "affinity_load_override"
    await pool.release(three, successful=True)
    assert pool.affinity.get("hot").worker == seed.worker.name
    for lease in [one, two]:
        await pool.release(lease)

    pool = await prefix_pool(capacity=1, affinity_max_skew=100)
    seed = await pool.acquire("hot")
    await pool.release(seed, successful=True)
    full = await pool.acquire("hot")
    other = await pool.acquire("hot")
    assert full.worker is not other.worker  # Even unlimited skew cannot exceed capacity.
    await pool.release(full)
    await pool.release(other)


async def test_health_invalidation_and_old_completion_cannot_rewarm_restarted_worker():
    pool = await prefix_pool()
    old = await pool.acquire("hot")
    await pool.set_health(old.worker, False)
    await pool.set_health(old.worker, True)
    await pool.release(old, successful=True)
    assert pool.affinity.get("hot") is None
    fresh = await pool.acquire("hot")
    await pool.release(fresh, successful=True)
    await pool.set_health(fresh.worker, False)
    assert pool.affinity.get("hot") is None
    fallback = await pool.acquire("hot")
    assert fallback.worker is not fresh.worker
    await pool.release(fallback)


async def test_cancelling_tokenization_does_not_release_its_actual_thread_permit():
    started, finish = threading.Event(), threading.Event()

    class SlowEncoder:
        def key(self, messages):
            started.set()
            finish.wait(2)
            return "key"

    preparer = PrefixPreparer(SlowEncoder(), capacity=1)
    waiting = asyncio.create_task(preparer.key([]))
    try:
        await asyncio.to_thread(started.wait, 2)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        with pytest.raises(AdmissionError, match="prefix_preparation_full"):
            await preparer.key([])
        assert len(preparer.tasks) == 1
    finally:
        finish.set()
        await preparer.close()

    async def drained():
        return not preparer.tasks

    await eventually(drained)
    assert await preparer.key([]) == "key"
    await preparer.close()


def test_local_huggingface_tokenizer_roundtrip_without_network(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                "[UNK]": 0,
                "user": 1,
                "assistant": 2,
                "hello": 3,
                "world": 4,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template="{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}",
    )
    tokenizer.save_pretrained(tmp_path)
    encoder = PrefixEncoder.from_pretrained("real", "fixed-revision", 2, str(tmp_path))
    data = [Message(role="user", content="hello world")]
    assert encoder.tokenize(data) == [1, 3, 4, 2]
    assert encoder.key(data) == encoder.key([Message(role="user", content="hello")])


async def test_queued_affinity_does_not_wait_for_its_full_home_when_other_replica_frees():
    pool = await prefix_pool(capacity=1)
    warm = await pool.acquire("hot")
    await pool.release(warm, successful=True)
    home = await pool.acquire("hot")
    other = await pool.acquire("hot")
    waiting = asyncio.create_task(pool.acquire("hot"))
    await asyncio.sleep(0.01)
    await pool.release(other)
    moved = await waiting
    assert moved.worker is other.worker
    assert moved.decision == "affinity_load_override"
    await pool.release(home)
    await pool.release(moved)

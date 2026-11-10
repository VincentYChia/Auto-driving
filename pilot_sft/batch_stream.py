"""Training micro-batches in the synchronous loop's order, optionally built ahead in threads (``--workers``).

``train_lora.py`` makes each training micro-batch in two steps:

- **draw** (``Draws``): the next ``batch`` positions of the shuffled order, after appending a new permutation
  of the training set whenever fewer than ``batch`` are left, then the ego-dropout flags (``Batches.draw``).
  These are training's only random choices (the ``random`` module and ``Batches.rng``). They are cheap and
  always made in the calling thread, one micro-batch after another.
- **encode** (``Batches.encode``): prompts, JPEG decode, the processor, labels. Pure, and the slow part.

``Prefetcher`` encodes in the calling thread (``workers=0``: the loop as it always was) or in ``workers``
threads, up to ``prefetch`` micro-batches ahead of the one in use, and hands them back strictly in draw
order. So what the model receives does not depend on ``workers``.

Each micro-batch comes with ``(order, random state, dropout-rng state)`` as they were right after it was
drawn. A checkpoint stores that, not the live state, which is ahead by what has been prefetched. A resumed
run then draws what the uninterrupted one would have, with any ``workers`` before or after the restart.

Stdlib only (no torch), so ``tests/data/test_pilot_batch_stream.py`` runs anywhere.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Deque, List, NamedTuple, Optional, Tuple


class Drawn(NamedTuple):
    """One micro-batch's random choices."""

    idx: List[int]  # positions in the training set
    masked: List[bool]  # per sample: ego state and past positions hidden (mask_ego)
    state: Tuple[List[int], Any, Any]  # (order, random state, dropout-rng state) right after this draw


class Draws:
    """The training loop's draws, one micro-batch per call, as the synchronous loop has always made them.

    ``order`` is what is left of the current pass (a checkpoint's ``"order"``). ``perm`` supplies the passes'
    permutations and its state: the ``random`` module in training (``random.sample``, ``random.getstate``).
    ``flags(idx)`` draws the ego-dropout flags from ``flags_rng`` (``Batches.draw`` and ``Batches.rng``).
    """

    def __init__(self, n: int, batch: int, order: List[int], perm: Any, flags: Callable[[List[int]], List[bool]],
                 flags_rng: Any):
        self.n, self.batch, self.order = n, batch, order
        self.perm, self.flags, self.flags_rng = perm, flags, flags_rng

    def __call__(self) -> Drawn:
        order = self.order
        if len(order) < self.batch:
            order = order + self.perm.sample(range(self.n), self.n)  # a new list: earlier states keep theirs
        idx, self.order = order[: self.batch], order[self.batch :]
        masked = self.flags(idx)
        return Drawn(idx, masked, (self.order, self.perm.getstate(), self.flags_rng.getstate()))


class Prefetcher:
    """``(drawn, encoded)`` per micro-batch from ``draw()`` and ``encode(drawn, tools)``, strictly in draw order.

    With ``workers=0`` both run in the calling thread, one micro-batch per ``next()``, and ``tools`` is None.
    With ``workers > 0`` the draws stay in the calling thread, in order, and ``workers`` threads encode up to
    ``prefetch`` (default ``2 * workers``) drawn micro-batches while the caller uses the current one. Each
    thread gets its own ``tools``, made by ``make_tools()`` (required with workers) here, in the calling
    thread: its own processor and image reader, since a fast tokenizer used from two threads fails ("Already
    borrowed") and ``ViewImages`` keeps one zip handle. Making a Prefetcher draws nothing and starts no thread,
    so it can be made before a timer starts and entered later. ``close()``, or leaving a ``with`` block, drops
    what was drawn ahead and joins the threads, also when the loop stops on an error.
    """

    def __init__(self, draw: Callable[[], Drawn], encode: Callable[[Drawn, Any], Any], workers: int = 0,
                 prefetch: Optional[int] = None, make_tools: Optional[Callable[[], Any]] = None):
        prefetch = 2 * workers if prefetch is None else prefetch
        if workers < 0 or (workers and prefetch < 1):
            raise ValueError(f"need workers >= 0 and, with workers, prefetch >= 1; got {workers} and {prefetch}")
        if workers and make_tools is None:  # else every thread would share the caller's processor and reader
            raise ValueError("workers need make_tools: each encode thread gets its own processor and reader")
        self.draw, self.encode, self.prefetch = draw, encode, prefetch
        self.pending: Deque[Tuple[Drawn, Future]] = deque()
        self.closed = False
        self.pool: Optional[ThreadPoolExecutor] = None
        if workers:
            self.spare: queue.SimpleQueue = queue.SimpleQueue()  # one set of tools per thread, bound on first use
            for _ in range(workers):
                self.spare.put(make_tools())
            self.local = threading.local()
            self.pool = ThreadPoolExecutor(workers, thread_name_prefix="encode")

    def __iter__(self) -> "Prefetcher":
        return self

    def __next__(self) -> Tuple[Drawn, Any]:
        if self.closed:
            raise RuntimeError("this Prefetcher is closed")
        if self.pool is None:
            drawn = self.draw()
            return drawn, self.encode(drawn, None)
        while len(self.pending) <= self.prefetch:  # the micro-batch handed out next, and `prefetch` behind it
            drawn = self.draw()
            self.pending.append((drawn, self.pool.submit(self._encode, drawn)))
        drawn, future = self.pending.popleft()
        return drawn, future.result()

    def _encode(self, drawn: Drawn) -> Any:
        if not hasattr(self.local, "tools"):
            self.local.tools = self.spare.get_nowait()
        return self.encode(drawn, self.local.tools)

    def close(self) -> None:
        self.closed = True
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()

    def __enter__(self) -> "Prefetcher":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

"""``pilot_sft/batch_stream.py``: the training micro-batches, and a resume, are the same for any ``--workers``.

The stream tests need only Pillow: a fake processor and a tiny view of generated JPEGs, loose or zipped.
The tests of ``train_lora``'s own ``Batches`` and ``train_stream`` need torch and are skipped without it.
They compare against the code as it was before the stream (copied below), so "unchanged" means unchanged.
"""

import contextlib
import copy
import io
import pickle
import random
import sys
import threading
import time
import types
import zipfile
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft"))

import batch_stream  # noqa: E402
import view_images  # noqa: E402

N_SAMPLES, BATCH = 10, 4  # micro-batches straddle passes: 10 samples are 2.5 micro-batches


@pytest.fixture(autouse=True)
def global_random():
    """The draws use the ``random`` module itself, as training does; leave it as it was."""
    state = random.getstate()
    yield
    random.setstate(state)


def make_items(n):
    return [{"id": f"s{k}", "image": f"images/log{k % 3}/tok{k}.jpg", "system": "You plan the path.",
             "prompt": f"Front camera image.\nEgo state: speed {k}.0 m/s.\nPast positions: ({-k}.0, 0.0).\n"
                       "Route command: go straight.", "target": f"({k}.0, 0.{k})"} for k in range(n)]


@pytest.fixture(params=["loose", "zip"])
def view(request, tmp_path):
    """(view directory, items): N_SAMPLES generated JPEGs, all different, as loose files or in images.zip."""
    items = make_items(N_SAMPLES)
    root = tmp_path / "view"
    blobs = {}
    for k, it in enumerate(items):
        picture = Image.new("RGB", (32, 18), (25 * k, 250 - 25 * k, 9 * k))
        picture.putpixel((k, k), (255, 255, 255))
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=90)
        blobs[it["image"]] = buffer.getvalue()
    if request.param == "loose":
        for rel, data in blobs.items():
            path = root.joinpath(*rel.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    else:
        root.mkdir()
        with zipfile.ZipFile(root / view_images.ZIP_NAME, "w", zipfile.ZIP_STORED) as archive:
            for rel in sorted(blobs):
                archive.writestr(rel, blobs[rel])
    return str(root), items


class OneThreadAtATime:
    """Fails as a fast tokenizer can ("Already borrowed") when two threads use it at once; records its threads."""

    def __init__(self):
        self.busy, self.threads = threading.Lock(), set()

    @contextlib.contextmanager
    def claim(self):
        if not self.busy.acquire(blocking=False):
            raise RuntimeError("Already borrowed")
        try:
            self.threads.add(threading.get_ident())
            time.sleep(0.001)  # hold it long enough for sharing to show
            yield
        finally:
            self.busy.release()


class FakeProcessor(OneThreadAtATime):
    """One id per character, and the decoded image's pixels."""

    def __call__(self, text, images):
        with self.claim():
            return {"input_ids": [[ord(c) for c in t] for t in text], "pixels": [im.tobytes() for im in images]}


class Tools:
    """One thread's processor and image reader (``train_lora``'s ``thread_tools``), recording the reader's threads."""

    def __init__(self, view):
        self.processor, self.images, self.threads = FakeProcessor(), view_images.ViewImages(view), set()

    def encode(self, items):
        self.threads.add(threading.get_ident())
        pictures = [self.images.open(it["image"]).convert("RGB") for it in items]
        return self.processor(text=[it["prompt"] + it["target"] for it in items], images=pictures)


def hide_ego(item):  # stands in for train_lora.mask_ego
    return {**item, "prompt": item["prompt"].replace("Ego state:", "Ego state: not available.")}


class Harness:
    """``batch_stream`` wired as ``train_lora.train_stream`` wires it, around the fake processor."""

    def __init__(self, view, items, share=0.5, seed=0):
        self.view, self.items, self.share = view, items, share
        random.seed(seed)  # train_lora: random.seed(args.seed)
        self.rng = random.Random(seed + 1)  # train_lora: Batches.rng
        self.main = Tools(view)  # the training loop's own processor and reader
        self.made = []  # tools made for encode threads
        self.draw_threads = set()

    def flags(self, idx):  # Batches.draw(items, train=True)
        self.draw_threads.add(threading.get_ident())
        return [self.rng.random() < self.share for _ in idx] if self.share > 0 else [False] * len(idx)

    def make_tools(self):
        self.made.append(Tools(self.view))
        return self.made[-1]

    def encode(self, drawn, tools):
        rows = [hide_ego(self.items[i]) if m else self.items[i] for i, m in zip(drawn.idx, drawn.masked)]
        return (tools or self.main).encode(rows)

    def draws(self, order=None):  # the list itself, as train_stream passes a checkpoint's
        return batch_stream.Draws(len(self.items), BATCH, [] if order is None else order, random, self.flags,
                                  self.rng)

    def stream(self, workers, order=None, prefetch=None):
        return batch_stream.Prefetcher(self.draws(order), self.encode, workers, prefetch, self.make_tools)


def take(stream, k):
    """k micro-batches as (idx, masked, encoded, state), with a copy of each state taken when it arrived."""
    out, copies = [], []
    for _ in range(k):
        drawn, encoded = next(stream)
        out.append((drawn.idx, drawn.masked, encoded, drawn.state))
        copies.append(copy.deepcopy(drawn.state))
    assert [row[3] for row in out] == copies  # no later draw changed a state already handed out
    return out


def encode_threads():
    return [t for t in threading.enumerate() if t.name.startswith("encode")]


def old_loop_draws(n, batch, share, k, seed=0):
    """The draws of train_lora's loop before batch_stream (copied), and the state its checkpoint saved."""
    random.seed(seed)
    rng = random.Random(seed + 1)
    order, out = [], []
    for _ in range(k):
        if len(order) < batch:
            order += random.sample(range(n), n)
        idx, order = order[:batch], order[batch:]
        masked = [rng.random() < share for _ in idx] if share > 0 else [False] * len(idx)  # the old collate's draws
        out.append((idx, masked, (list(order), random.getstate(), rng.getstate())))
    return out


@pytest.mark.parametrize("n, batch", [(10, 4), (8, 4), (3, 4), (1, 1)])
@pytest.mark.parametrize("share", [0.0, 0.5])
def test_draws_are_the_old_loops_draws_and_checkpoint_state(n, batch, share):
    expected = old_loop_draws(n, batch, share, 40)
    random.seed(0)
    rng = random.Random(1)
    flags = lambda idx: [rng.random() < share for _ in idx] if share > 0 else [False] * len(idx)  # noqa: E731
    draws = batch_stream.Draws(n, batch, [], random, flags, rng)
    got = [tuple(draws()) for _ in range(40)]
    assert got == expected
    if n < batch:  # one new pass per micro-batch at most, as before: the micro-batch comes out short
        assert {len(idx) for idx, _, _ in got} == {n}


@pytest.mark.parametrize("workers, prefetch", [(3, None), (3, 1), (2, 7)])
def test_workers_do_not_change_the_micro_batches(view, workers, prefetch):
    path, items = view
    sync = Harness(path, items)
    with sync.stream(0) as stream:
        expected = take(stream, 25)  # 100 samples: 10 passes
    ahead = Harness(path, items)
    with ahead.stream(workers, prefetch=prefetch) as stream:
        got = take(stream, 25)
    assert got == expected  # indices, dropout flags, encoded outputs, and the states a checkpoint would save
    flags = [m for _, masked, _, _ in got for m in masked]
    assert any(flags) and not all(flags)
    assert ahead.draw_threads == {threading.get_ident()}  # the draws stay in the calling thread
    assert sync.main.threads == sync.main.processor.threads == {threading.get_ident()} and not sync.made
    # each encode thread used its own processor and reader, and the loop's own were left alone
    assert len(ahead.made) == workers and not ahead.main.threads and not ahead.main.processor.threads
    used = [t for t in ahead.made if t.threads]
    assert used and all(len(t.threads) == 1 and t.processor.threads == t.threads for t in used)
    threads = set().union(*(t.threads for t in used))
    assert len(threads) == len(used) and threading.get_ident() not in threads
    assert not encode_threads()  # joined on leaving the with block


def test_the_prefetcher_draws_exactly_prefetch_ahead(view):
    path, items = view
    for workers, prefetch, ahead in ((0, None, 0), (2, 3, 3), (3, None, 6)):
        h = Harness(path, items)
        draws, drawn = h.draws(), []
        count = lambda: drawn.append(draws()) or drawn[-1]  # noqa: E731
        with batch_stream.Prefetcher(count, h.encode, workers, prefetch, h.make_tools) as stream:
            for k in range(1, 8):
                next(stream)
                assert len(drawn) == k + ahead


@pytest.mark.parametrize("before, after", [(0, 3), (3, 0), (3, 3), (0, 0)])
@pytest.mark.parametrize("stop", [1, 5, 12])  # mid-pass; at a pass end (order empty); short remainder
def test_resume_from_the_saved_state_continues_as_an_uninterrupted_run(view, before, after, stop):
    path, items = view
    total = 20
    whole = Harness(path, items)
    with whole.stream(0) as stream:
        expected = take(stream, total)

    first = Harness(path, items)
    with first.stream(before) as stream:
        head = take(stream, stop)
        live = (random.getstate(), first.rng.getstate())
    order, py_rng, dropout_rng = pickle.loads(pickle.dumps(head[-1][3]))  # through a checkpoint file
    assert (live == (py_rng, dropout_rng)) == (before == 0)  # with workers the live state is ahead: not saved
    second = Harness(path, items)  # a new process: seeded as at the start, then the checkpoint restored
    random.setstate(py_rng)
    second.rng.setstate(dropout_rng)
    with second.stream(after, order=order) as stream:
        tail = take(stream, total - stop)
    assert head + tail == expected


@pytest.mark.parametrize("stop", [2, 5, 7])
def test_a_checkpoint_of_the_old_loop_resumes_with_workers(view, stop):
    path, items = view
    expected = old_loop_draws(N_SAMPLES, BATCH, 0.5, 16)
    order, py_rng, dropout_rng = expected[stop - 1][2]  # what the old save_checkpoint stored after `stop`
    h = Harness(path, items)
    random.setstate(py_rng)
    h.rng.setstate(dropout_rng)
    with h.stream(3, order=order) as stream:
        tail = take(stream, 16 - stop)
    assert [(idx, masked, state) for idx, masked, _, state in tail] == expected[stop:]


def test_the_checkpoint_state_has_todays_types(view):
    path, items = view
    h = Harness(path, items)
    start = [3, 1]
    with h.stream(3, order=start) as stream:
        got = take(stream, 6)
    assert start == [3, 1]  # the order handed in (a checkpoint's) is not changed either
    todays = random.getstate(), random.Random(1).getstate()  # what save_checkpoint stored for py_rng, dropout_rng
    for _, _, _, (order, py_rng, dropout_rng) in got:
        assert type(order) is list and all(type(i) is int for i in order)
        for ours, theirs in ((py_rng, todays[0]), (dropout_rng, todays[1])):
            assert type(ours) is tuple and len(ours) == len(theirs) == 3 and ours[0] == theirs[0]
            assert type(ours[1]) is tuple and len(ours[1]) == len(theirs[1]) and {type(x) for x in ours[1]} == {int}
            assert type(ours[2]) is type(theirs[2])
            random.Random().setstate(ours)


def test_a_failed_encode_surfaces_at_its_own_micro_batch_and_the_threads_stop(view):
    path, items = view
    items = items[:-1] + [{**items[-1], "image": "images/no_log/no_token.jpg"}]

    def run(workers, prefetch=None):
        h = Harness(path, items)
        delivered = []
        with pytest.raises(FileNotFoundError):
            with h.stream(workers, prefetch=prefetch) as stream:
                while True:
                    delivered.append(next(stream)[0].idx)
        return delivered

    expected = run(0)
    assert run(3) == run(2, 9) == expected  # the error comes when its micro-batch is due, not when it was encoded
    assert not encode_threads()
    h = Harness(*view)
    stream = h.stream(2)
    next(stream)
    stream.close()
    stream.close()
    with pytest.raises(RuntimeError, match="closed"):
        next(stream)
    with pytest.raises(ValueError, match="prefetch"):
        batch_stream.Prefetcher(h.draws(), h.encode, workers=2, prefetch=0, make_tools=h.make_tools)


def test_workers_without_their_own_tools_are_refused(view):
    h = Harness(*view)
    with pytest.raises(ValueError, match="make_tools"):  # else every thread would share the loop's processor
        batch_stream.Prefetcher(h.draws(), h.encode, workers=2)
    with batch_stream.Prefetcher(h.draws(), h.encode, workers=0) as stream:  # in the loop: the loop's own
        next(stream)
    assert h.main.threads == {threading.get_ident()}


def test_making_the_stream_loads_the_tools_and_draws_nothing(view):
    """train_lora makes the stream before its clock starts, and enters it later: only the tools load then."""
    path, items = view
    sync = Harness(path, items)
    with sync.stream(0) as stream:
        expected = take(stream, 8)
    h = Harness(path, items)
    before = random.getstate(), h.rng.getstate()
    stream = h.stream(3)
    assert len(h.made) == 3 and (random.getstate(), h.rng.getstate()) == before  # nothing drawn yet
    assert not encode_threads()  # nor a thread started: nothing to leak if training stops before the loop
    with stream:
        assert take(stream, 8) == expected
    assert not encode_threads()


# --- train_lora's own Batches and train_stream (torch) ---------------------------------------------------------


class TorchFakeProcessor(OneThreadAtATime):
    """The Qwen processor's interface as ``Batches`` uses it, returning torch tensors."""

    def __init__(self):
        super().__init__()
        self.tokenizer = types.SimpleNamespace(padding_side="right", eos_token="<eos>",
                                               get_vocab=lambda: {"<|im_end|>": 1})

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "<s>" + messages[0]["content"][0]["text"] + "<u>" + messages[1]["content"][1]["text"] + "<a>"

    def __call__(self, text, images, padding=True, return_tensors=None):
        import torch

        with self.claim():
            ids = [[sum(im.tobytes()) % 997 + 2] + [ord(c) % 1000 + 2 for c in t] for t, im in zip(text, images)]
            if return_tensors is None:
                return {"input_ids": ids}
            width = max(map(len, ids))
            return {"input_ids": torch.tensor([i + [0] * (width - len(i)) for i in ids]),
                    "attention_mask": torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i in ids]),
                    "pixel_values": torch.stack([torch.tensor(list(im.tobytes()), dtype=torch.uint8) for im in images])}


def old_collate(self, items, train=False):
    """``Batches.collate`` before the split into draw and encode (copied)."""
    import train_lora

    if train and self.ego_dropout > 0:
        items = [train_lora.mask_ego(it) if self.rng.random() < self.ego_dropout else it for it in items]
    prompts = [train_lora.prompt_text(self.processor, it) for it in items]
    texts = [p + it["target"] + self.end for p, it in zip(prompts, items)]
    images = [self.images.open(it["image"]).convert("RGB") for it in items]
    batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
    prompt_ids = self.processor(text=prompts, images=images, padding=False)["input_ids"]
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    for row, ids in enumerate(prompt_ids):
        n_prompt = len(ids)
        pad = int((batch["attention_mask"][row] == 0).sum()) if self.processor.tokenizer.padding_side == "left" else 0
        labels[row, : pad + n_prompt] = -100
    batch["labels"] = labels
    return batch


def same_tensors(a, b):
    import torch

    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_collate_is_draw_then_encode_and_unchanged(view):
    pytest.importorskip("torch")
    import train_lora

    path, items = view
    old, new = (train_lora.Batches(TorchFakeProcessor(), path, items, ego_dropout=0.5, seed=3) for _ in range(2))
    masked = []
    for chunk in (items[:4], items[4:8], items[8:], items[:4]):
        flags = new.draw(chunk, train=True)
        masked += flags
        assert same_tensors(old_collate(old, chunk, train=True), new.encode(chunk, flags))
        assert old.rng.getstate() == new.rng.getstate()
        assert same_tensors(old_collate(old, chunk, train=True), new.collate(chunk, train=True))
    assert any(masked) and not all(masked)
    state = new.rng.getstate()
    assert new.draw(items[:4]) == [False] * 4 and new.rng.getstate() == state  # dev batches draw nothing
    assert same_tensors(old_collate(old, items[:4]), new.collate(items[:4])) and new.rng.getstate() == state
    labels = new.collate(items[:2])["labels"]
    assert (labels != -100).any() and (labels[:, 0] == -100).all()  # the loss covers the answer only


def old_training_batches(path, items, k):
    """The training loop's micro-batches before batch_stream (copied), and its checkpoint state after each."""
    import train_lora

    random.seed(0)
    batches = train_lora.Batches(TorchFakeProcessor(), path, items, ego_dropout=0.5, seed=0)
    order, out = [], []
    for _ in range(k):
        if len(order) < BATCH:
            order += random.sample(range(len(items)), len(items))
        idx, order = order[:BATCH], order[BATCH:]
        out.append((idx, old_collate(batches, [items[i] for i in idx], train=True),
                    (list(order), random.getstate(), batches.rng.getstate())))
    return out


def new_training_batches(path, items, k, workers, order=(), state=None):
    import train_lora

    random.seed(0)
    batches = train_lora.Batches(TorchFakeProcessor(), path, items, ego_dropout=0.5, seed=0)
    if state is not None:  # as main() restores a checkpoint
        random.setstate(state["py_rng"])
        batches.rng.setstate(state["dropout_rng"])
    made = []

    def thread_tools():
        made.append(TorchFakeProcessor())
        return made[-1], view_images.ViewImages(path)

    with train_lora.train_stream(batches, items, list(order), BATCH, workers, None, thread_tools) as stream:
        out = [(d.idx, encoded, d.state) for d, encoded in (next(stream) for _ in range(k))]
    assert len(made) == workers and not encode_threads()
    assert batches.processor.threads == (set() if workers else {threading.get_ident()})
    assert all(len(p.threads) <= 1 for p in made)
    return out


@pytest.mark.parametrize("workers", [0, 3])
def test_train_stream_gives_the_old_loops_tensors(view, workers):
    pytest.importorskip("torch")
    path, items = view
    expected = old_training_batches(path, items, 15)
    got = new_training_batches(path, items, 15, workers)
    assert [(i, s) for i, _, s in got] == [(i, s) for i, _, s in expected]
    assert all(same_tensors(a, b) for (_, a, _), (_, b, _) in zip(got, expected))


def test_train_stream_refuses_workers_without_their_own_tools(view):
    pytest.importorskip("torch")
    import train_lora

    path, items = view
    batches = train_lora.Batches(TorchFakeProcessor(), path, items, ego_dropout=0.5, seed=0)
    with pytest.raises(ValueError, match="make_tools"):
        train_lora.train_stream(batches, items, [], BATCH, workers=2)


@pytest.mark.parametrize("before, after", [(3, 0), (0, 3)])
def test_train_stream_resumes_through_a_torch_checkpoint(view, before, after):
    torch = pytest.importorskip("torch")
    path, items = view
    expected = old_training_batches(path, items, 15)
    head = new_training_batches(path, items, 6, before)
    order, py_rng, dropout_rng = head[-1][2]
    buffer = io.BytesIO()  # the keys and values save_checkpoint writes for the draws
    torch.save({"order": order, "py_rng": py_rng, "dropout_rng": dropout_rng}, buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=False)
    tail = new_training_batches(path, items, 9, after, order=state["order"], state=state)
    assert [i for i, _, _ in head + tail] == [i for i, _, _ in expected]
    assert all(same_tensors(a, b) for (_, a, _), (_, b, _) in zip(head + tail, expected))

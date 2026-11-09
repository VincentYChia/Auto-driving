"""The packed pilot view (``pilot_view build --storage zip``) and its readers (``pilot_sft/view_images.py``)."""

import hashlib
import io
import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")
Image = pytest.importorskip("PIL.Image")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft"))

import view_images  # noqa: E402
from data_pipeline.v1 import canonical, pilot_view  # noqa: E402
from tests.data.test_v1_canonical import fetched  # noqa: E402,F401  (fixture)


def lines(view, split):
    return (view / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()


def images_of(view):
    return [json.loads(line)["image"] for split in ("train", "dev") for line in lines(view, split)]


@pytest.fixture
def both_views(fetched, tmp_path):  # noqa: F811
    """The smoke view built loose (moved to ``loose/``) and then zipped in place: (out, loose, packed, specs)."""
    out, metadata = fetched
    canonical.build(str(out), ("smoke",), str(metadata))
    packed = Path(pilot_view.view_dir(str(out), "smoke"))
    loose_spec = pilot_view.build(str(out), "smoke", width=16, workers=1)
    loose = tmp_path / "loose"
    shutil.move(str(packed), str(loose))
    zip_spec = pilot_view.build(str(out), "smoke", width=16, workers=1, storage="zip")
    return out, loose, packed, loose_spec, zip_spec


def test_zip_view_holds_the_loose_views_bytes_and_the_same_jsonl(both_views):
    out, loose, packed, loose_spec, zip_spec = both_views
    for split in ("train", "dev"):
        assert lines(packed, split) == lines(loose, split) and lines(packed, split)
    assert not (packed / "images").exists() and not (loose / pilot_view.ZIP_NAME).exists()
    names = images_of(loose)
    with zipfile.ZipFile(packed / pilot_view.ZIP_NAME) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == sorted(names)  # member name = the jsonl "image" field, sorted
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED and info.date_time == pilot_view.ZIP_DATE
            assert archive.read(info) == (loose / info.filename).read_bytes()
    assert (loose_spec["storage"], loose_spec["images_zip"]) == ("loose", None)
    meta = zip_spec["images_zip"]
    data = (packed / pilot_view.ZIP_NAME).read_bytes()
    assert zip_spec["storage"] == "zip" and meta["members"] == len(names) == zip_spec["image"]["resized_now"]
    assert meta["bytes"] == len(data) and meta["sha256"] == hashlib.sha256(data).hexdigest()
    assert zip_spec["image"]["measured_sizes"] == loose_spec["image"]["measured_sizes"] == {"16x9": len(names)}
    assert zip_spec["image"]["height"] == 9 and json.loads((packed / "view.json").read_text())["storage"] == "zip"
    assert not list(packed.glob("*.part"))

    again = pilot_view.build(str(out), "smoke", width=16, workers=2, storage="zip")
    assert again["images_zip"]["sha256"] == meta["sha256"]  # deterministic, whatever the worker count


def test_chat_export_is_the_same_from_either_layout(both_views):
    out, loose, packed, _, _ = both_views
    exported = pilot_view.export(str(out), "smoke", limit=3)
    for split in ("train", "dev"):
        loose_lines = []
        for line in lines(loose, split)[:3]:
            item = json.loads(line)
            loose_lines.append(json.dumps(pilot_view.chat_record(item, (loose / item["image"]).read_bytes())))
        assert (packed / f"chat_{split}.jsonl").read_text().splitlines() == loose_lines
        assert exported[split]["samples"] == len(loose_lines)


def test_view_images_reads_both_layouts(both_views):
    _, loose, packed, _, _ = both_views
    from_zip, from_files = view_images.ViewImages(str(packed)), view_images.ViewImages(str(loose))
    assert from_zip.packed and not from_files.packed
    for rel in images_of(loose):
        assert from_zip.read(rel) == from_files.read(rel) == (loose / rel).read_bytes()
    rel = images_of(loose)[0]
    assert from_zip.open(rel).size == from_files.open(rel).size == (16, 9)
    clone = pickle.loads(pickle.dumps(from_zip))  # what a spawned data-loader worker gets
    assert clone.read(rel) == from_zip.read(rel)
    with pytest.raises(FileNotFoundError):
        from_zip.read("images/no_log/no_token.jpg")
    from_zip.close()
    assert from_zip.read(rel) == from_files.read(rel)  # re-opens after close


def test_resize_bytes_is_what_the_file_wrapper_writes(tmp_path):
    src = tmp_path / "src.jpg"
    Image.new("RGB", (64, 36), (10, 200, 30)).save(src, format="JPEG")
    dst = tmp_path / "out" / "dst.jpg"
    assert pilot_view._resize((str(src), str(dst), 16)) == 1 and pilot_view._resize((str(src), str(dst), 16)) == 0
    assert dst.read_bytes() == pilot_view.resize_bytes(src.read_bytes(), 16)
    assert Image.open(dst).size == (16, 9)
    with pytest.raises(ValueError):
        pilot_view.write_zip(str(tmp_path / "x.zip"), ["b", "a"], [None, None], workers=1)


class StubTokenizer:
    padding_side = "right"
    eos_token = "<eos>"

    def get_vocab(self):
        return {"<|im_end|>": 1}


class StubProcessor:
    """Character-level stand-in for a Qwen processor: enough for ``Batches.collate`` on CPU."""

    tokenizer = StubTokenizer()

    def __init__(self):
        self.images = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "<prompt>" + messages[1]["content"][1]["text"] + "<answer>"

    def __call__(self, text, images, padding=True, return_tensors=None):
        import torch

        self.images.append(images)
        ids = [[ord(c) % 1000 + 2 for c in t] for t in text]
        if return_tensors is None:
            return {"input_ids": ids}
        width = max(map(len, ids))
        return {"input_ids": torch.tensor([i + [0] * (width - len(i)) for i in ids]),
                "attention_mask": torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i in ids])}


def test_train_lora_collate_loads_images_from_the_zip(both_views):
    pytest.importorskip("torch")
    import train_lora

    _, loose, packed, _, _ = both_views
    items = [json.loads(line) for line in lines(packed, "train")[:2]]
    processor = StubProcessor()
    batch = train_lora.Batches(processor, str(packed), items).collate(items)
    assert batch["labels"].shape == batch["input_ids"].shape and (batch["labels"] != -100).any()
    images = processor.images[0]
    assert [im.mode for im in images] == ["RGB", "RGB"] and images[0].size == (16, 9)
    for im, item in zip(images, items):
        expected = Image.open(io.BytesIO((loose / item["image"]).read_bytes())).convert("RGB")
        assert im.tobytes() == expected.tobytes()


def test_packed_originals_to_zip_view_end_to_end(tmp_path):
    """fetch --storage pack -> canonical -> pilot_view build --storage zip -> ViewImages, against the loose path."""
    from data_pipeline.v1 import image_store
    from tests.data.test_v1_canonical import build_dataset

    views = {}
    try:
        for storage in ("loose", "pack"):
            out, metadata = build_dataset(tmp_path / storage, storage=storage)
            assert (out / "images" / "CAM_F0").exists() == (storage == "loose")
            assert bool(list((out / "images").glob("packs/*.zip"))) == (storage == "pack")
            report = canonical.build(str(out), ("smoke",), str(metadata))
            assert [c["check"] for c in report["checks"] if not c["ok"]] == []
            image_store.close_all()  # Windows cannot delete an open pack
            spec = pilot_view.build(str(out), "smoke", width=16, workers=2, storage="zip")
            views[storage] = (Path(pilot_view.view_dir(str(out), "smoke")), spec)
    finally:
        image_store.close_all()
    (loose_view, loose_spec), (pack_view, pack_spec) = views["loose"], views["pack"]
    assert pack_spec["images_zip"]["sha256"] == loose_spec["images_zip"]["sha256"]  # same view bytes from packs
    for split in ("train", "dev"):
        assert lines(pack_view, split) == lines(loose_view, split) and lines(pack_view, split)
    images = view_images.ViewImages(str(pack_view))
    assert images.packed and not (pack_view / "images").exists()
    with zipfile.ZipFile(loose_view / pilot_view.ZIP_NAME) as archive:
        for rel in images_of(pack_view):
            assert images.read(rel) == archive.read(rel)
            assert images.open(rel).size == (16, 9)
    images.close()


def test_a_loose_build_refuses_a_view_folder_that_holds_images_zip(both_views):
    out, _, packed, _, zip_spec = both_views
    with pytest.raises(FileExistsError, match="images.zip"):  # the readers would serve the zip, not these images
        pilot_view.build(str(out), "smoke", width=24, workers=1)
    assert json.loads((packed / "view.json").read_text(encoding="utf-8"))["storage"] == "zip"
    assert not (packed / "images").exists()


def test_a_member_missing_from_its_pack_names_the_pack_and_leaves_no_part(tmp_path):
    from data_pipeline.v1 import image_store, tables
    from tests.data.test_v1_canonical import build_dataset

    out, metadata = build_dataset(tmp_path, storage="pack")
    canonical.build(str(out), ("smoke",), str(metadata))
    frames = tables.read(str(out / "frames.parquet"), ["image_path", "source_archive", "in_smoke"])
    k = int(next(i for i, v in enumerate(frames["in_smoke"]) if v))
    victim, archive = str(frames["image_path"][k]), int(frames["source_archive"][k])
    path = image_store.pack_path(str(out), archive)
    with zipfile.ZipFile(path) as z:
        keep = [(n, z.read(n)) for n in z.namelist() if n != victim]
    image_store.close_all()
    image_store.write_pack(path, keep)
    with pytest.raises(FileNotFoundError, match=rf"is not in .*{archive}\.zip \(archive {archive}\)"):
        pilot_view.build(str(out), "smoke", width=16, workers=1, storage="zip")
    image_store.close_all()
    view = Path(pilot_view.view_dir(str(out), "smoke"))
    assert not list(view.glob("images.zip*"))

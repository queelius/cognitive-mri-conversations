import io
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pyarrow.parquet as pq
import pytest

from acquire import RangeReader, get_bytes
from audit import KEY, unique_label_index, join_labels, fingerprint, row_key


class Response(io.BytesIO):
    def __init__(self, data=b"abc", status=206, content_range="bytes 0-2/10"):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}


def test_ranges_are_bounded_and_hashed():
    reader = RangeReader("https://example.org/data", 10, 3)
    with patch("urllib.request.urlopen", return_value=Response()) as request:
        assert reader.read(3) == b"abc"
        assert request.call_args.args[0].headers["Range"] == "bytes=0-2"
    assert reader.tell() == 3 and reader.downloaded == 3 and len(reader.reads[0]["sha256"]) == 64
    with pytest.raises(RuntimeError, match="cap"):
        reader.read(1)
    assert reader.seek(-1, 2) == 9
    assert reader.seek(-3, 1) == 6
    assert reader.seek(10) == 10 and reader.read() == b""
    assert reader.readable() and reader.seekable()
    with pytest.raises(ValueError):
        reader.seek(-1)


@pytest.mark.parametrize("response", [Response(status=200), Response(b"a"), Response(content_range="bytes 1-3/10")])
def test_range_errors_fail_closed(response):
    with patch("urllib.request.urlopen", return_value=response), pytest.raises(RuntimeError):
        RangeReader("https://example.org/data", 10, 10).read(3)


def test_full_download_cap():
    with patch("urllib.request.urlopen", return_value=Response(b"abcd")), pytest.raises(RuntimeError, match="cap"):
        get_bytes("https://example.org/data", 3)
    with patch("urllib.request.urlopen", return_value=Response(b"abc")):
        assert get_bytes("https://example.org/data", 3) == b"abc"


def test_public_source_join_integration():
    base = Path(__file__).parent
    if not (base / "cache/content-sample.parquet").exists():
        pytest.skip("Pinned public sample not cached")
    source = pq.read_table(base / "cache/content-sample.parquet", columns=list(KEY)).to_pylist()
    indexes = {}
    for kind, field in [("domain", "conversation_domain"), ("intent", "user_intent")]:
        rows = pq.read_table(base / f"cache/{kind}.parquet", columns=list(KEY) + [field]).to_pylist()
        indexes[field], stats = unique_label_index(rows, field)
        assert stats["duplicate_keys"] == 0
    merged = join_labels(source, indexes)
    assert len(merged) == 2000
    assert sum(all(r[k] is not None for k in indexes) for r in merged) >= 1990
    manifest = json.loads((base / "source-manifest.json").read_text())
    assert manifest["total_downloaded_record_bytes"] <= 60 * 1024**2


def test_saved_representation_integration():
    base = Path(__file__).parent
    if not (base / "cache/embeddings.npz").exists():
        pytest.skip("Representation check not run")
    manifest = json.loads((base / "split-manifest.json").read_text())
    assert len({r["record_id"] for r in manifest}) == len(manifest)
    assert len({r["component_id"] for r in manifest}) == len(manifest)
    dev = {r["component_id"] for r in manifest if r["split"] == "development"}
    evaluation = {r["component_id"] for r in manifest if r["split"] == "evaluation"}
    assert dev and evaluation and not dev.intersection(evaluation)
    saved = np.load(base / "cache/embeddings.npz")
    for role in saved.files:
        values = saved[role]
        assert values.shape == (len(manifest), 384)
        assert np.isfinite(values).all()
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5)


def test_saved_split_has_one_representative_per_full_source_component():
    """Rebuild the graph independently, without eligibility or language filtering."""
    from collections import defaultdict
    base = Path(__file__).parent
    if not (base / "cache/content-sample.parquet").exists():
        pytest.skip("Pinned public sample not cached")
    source = pq.read_table(base / "cache/content-sample.parquet").to_pylist()
    manifest = json.loads((base / "split-manifest.json").read_text())
    config = json.loads((base / "config.json").read_text())
    selected = {r["record_id"]: r["split"] for r in manifest}
    adjacency, token_owner = defaultdict(set), {}
    for i, row in enumerate(source):
        prompt = next((m["content"] for m in row["conversation"]
                       if m["role"] == "user" and (m.get("content") or "").strip()), "")
        prefix = " ".join(prompt.split())[:config["prompt_prefix_characters"]]
        for token in [("ip", row.get("hashed_ip")), ("prompt", prefix)]:
            if not token[1]:
                continue
            if token in token_owner:
                j = token_owner[token]
                adjacency[i].add(j)
                adjacency[j].add(i)
            else:
                token_owner[token] = i
    visited, selected_per_component, mixed_splits = set(), [], 0
    for i in range(len(source)):
        if i in visited:
            continue
        pending, splits = [i], []
        while pending:
            j = pending.pop()
            if j in visited:
                continue
            visited.add(j)
            identifier = fingerprint(row_key(source[j]))
            if identifier in selected:
                splits.append(selected[identifier])
            pending.extend(adjacency[j] - visited)
        selected_per_component.append(len(splits))
        mixed_splits += len(set(splits)) > 1
    assert sum(selected_per_component) == len(manifest)
    assert mixed_splits == 0, f"{mixed_splits} full-source components cross the split"
    assert max(selected_per_component) <= 1

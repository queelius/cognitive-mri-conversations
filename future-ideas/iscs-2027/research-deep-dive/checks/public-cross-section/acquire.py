"""Bounded, revision-pinned public data acquisition; no prompt text is printed."""
from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RangeReader(io.RawIOBase):
    """Seekable remote source with exact bounded Range requests, verified 206s."""

    def __init__(self, url: str, size: int, byte_cap: int):
        self.url, self.size, self.byte_cap = url, size, byte_cap
        self.position = 0
        self.downloaded = 0
        self.reads = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if self.position < 0:
            raise ValueError("Negative remote position")
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        if self.downloaded + size > self.byte_cap:
            raise RuntimeError("Predeclared byte cap exceeded")
        start, end = self.position, self.position + size - 1
        # A distinct query key prevents intermediary caches from reusing a different range.
        separator = "&" if "?" in self.url else "?"
        request = urllib.request.Request(self.url + separator + f"range={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError(f"Server ignored Range ({response.status}); full download refused")
            data = response.read(size + 1)
            if len(data) != size:
                raise RuntimeError("Partial response length disagrees with requested range")
            if response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise RuntimeError("Unexpected Content-Range")
        self.reads.append({"start": start, "end": end, "bytes": size, "sha256": digest(data)})
        self.downloaded += size
        self.position += size
        return data


def get_bytes(url, cap):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(cap + 1)
    if len(data) > cap:
        raise RuntimeError("Download larger than predeclared cap")
    return data


def main():
    start = time.perf_counter()
    config = json.loads((BASE / "config.json").read_text())
    cache = BASE / "cache"
    cache.mkdir(exist_ok=True)
    sources = []
    total = 0
    for kind in ["domain", "intent"]:
        path = f"wildchat-results/wildchat-4.8m-results/no-content/no-content-wildchat-4.8m-{config['annotation_shard']}-{kind}.parquet"
        revision = config["annotation_revision"]
        repo = config["annotation_repo"]
        url = f"https://media.githubusercontent.com/media/{repo}/{revision}/{path}"
        pointer_url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
        pointer = get_bytes(pointer_url, 1000).decode()
        fields = dict(line.split(" ", 1) for line in pointer.splitlines())
        expected_size = int(fields["size"])
        expected_sha = fields["oid"].split(":", 1)[1]
        target = cache / f"{kind}.parquet"
        data = target.read_bytes() if target.exists() else get_bytes(url, config["total_byte_cap"] - total)
        if len(data) != expected_size or digest(data) != expected_sha:
            raise RuntimeError("LFS size/hash mismatch")
        target.write_bytes(data)
        total += len(data)
        sources.append({"kind": kind, "url": url, "bytes": len(data), "sha256": digest(data), "lfs_pointer": pointer})
        table = pq.read_table(target)
        print(kind, "rows", len(table), "columns", table.column_names, flush=True)
    dataset, revision, path = config["dataset"], config["dataset_revision"], config["source_path"]
    info_url = f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}/data?limit=100"
    info = json.loads(get_bytes(info_url, 1000000))
    entry = next(x for x in info if x["path"] == path)
    url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"
    reader = RangeReader(url, entry["size"], min(config["content_byte_cap"], config["total_byte_cap"] - total))
    parquet = pq.ParquetFile(reader)
    row_groups = [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)]
    print("source", "rows", parquet.metadata.num_rows, "row_groups", row_groups, flush=True)
    # Selection was frozen as the first 2,000 physical rows; only load whole groups
    # needed to reach that cap, and enforce the transfer cap independently.
    chosen, seen = [], 0
    for i, n in enumerate(row_groups):
        if seen >= config["content_row_cap"]:
            break
        chosen.append(i)
        seen += n
    cols = ["conversation_hash", "model", "timestamp", "turn", "language", "hashed_ip", "state", "country", "conversation"]
    table = parquet.read_row_groups(chosen, columns=cols, use_threads=False).slice(0, config["content_row_cap"])
    pq.write_table(table, cache / "content-sample.parquet")
    sources.append({"kind": "content", "url": url, "source_metadata": entry, "source_rows": parquet.metadata.num_rows, "row_groups": row_groups, "selected_row_groups": chosen, "selected_rows": len(table), "downloaded_bytes": reader.downloaded, "ranges": reader.reads, "sample_sha256": digest((cache / "content-sample.parquet").read_bytes())})
    manifest = {"sources": sources, "total_downloaded_record_bytes": total + reader.downloaded, "elapsed_seconds": time.perf_counter() - start, "config_sha256": digest((BASE / "config.json").read_bytes()), "protocol_sha256": digest((BASE / "PROTOCOL.md").read_bytes())}
    (BASE / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sample_rows":len(table),"downloaded_record_bytes":manifest["total_downloaded_record_bytes"],"seconds":manifest["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

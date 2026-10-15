"""Public-slice eligibility and predeclared role/representation measurement check."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
KEY = ("conversation_hash", "model", "timestamp", "turn", "language", "hashed_ip", "state", "country")


def utc(timestamp):
    return timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp.astimezone(timezone.utc)


def day(timestamp):
    return utc(timestamp).date().isoformat()


def row_key(row):
    return tuple(utc(row[k]).isoformat() if k == "timestamp" else row.get(k) for k in KEY)


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def unique_label_index(rows, field):
    result, duplicates = {}, 0
    for row in rows:
        key, label = row_key(row), row.get(field)
        # The released classifier serializes failed/missing outcomes as "None".
        # This is a missing-value representation, not a substantive category.
        label = None if label == "None" else label
        if key in result:
            duplicates += 1
            if result[key] != label:
                raise ValueError("Conflicting labels for a complete composite key")
        result[key] = label
    return result, {"rows": len(rows), "unique_keys": len(result), "duplicate_keys": duplicates}


def join_labels(rows, indexes):
    return [{**r, **{field: index.get(row_key(r)) for field, index in indexes.items()}} for r in rows]


def eligibility(rows):
    hashes = Counter(r["conversation_hash"] for r in rows)
    proxy = defaultdict(list)
    for row in rows:
        if row.get("hashed_ip"):
            proxy[row["hashed_ip"]].append(row)
    profiles = []
    for group in proxy.values():
        profile = {"rows": len(group), "contents": len({r["conversation_hash"] for r in group}),
                   "days": len({day(r["timestamp"]) for r in group})}
        for field in ["country", "state", "language"]:
            profile[field] = len({r.get(field) for r in group})
        profile["screen_pass"] = profile["rows"] <= 161 and all(profile[k] <= 3 for k in ["country", "state", "language"])
        profiles.append(profile)
    result = {
        "physical_rows": len(rows), "unique_content_hashes": len(hashes),
        "duplicate_content_rows": len(rows) - len(hashes),
        "content_hashes_with_repeats": sum(n > 1 for n in hashes.values()),
        "maximum_content_multiplicity": max(hashes.values(), default=0),
        "proxy_count": len(proxy), "missing_proxy_rows": sum(not r.get("hashed_ip") for r in rows),
        "date_min": min(day(r["timestamp"]) for r in rows), "date_max": max(day(r["timestamp"]) for r in rows),
        "models": dict(Counter(r["model"] for r in rows)), "languages": dict(Counter(r["language"] for r in rows)),
        "proxy_content_quantiles": dict(zip(["min", "median", "p90", "p99", "max"], np.quantile([p["contents"] for p in profiles], [0, .5, .9, .99, 1]).tolist())),
        "single_content_proxies": sum(p["contents"] == 1 for p in profiles),
        "single_date_proxies": sum(p["days"] == 1 for p in profiles),
        "proxy_screen_failures": {"over_161_rows": sum(p["rows"] > 161 for p in profiles),
                                  **{f"over_3_{k}": sum(p[k] > 3 for p in profiles) for k in ["country", "state", "language"]}},
        "archive_thresholds": {}, "archive_thresholds_after_within_shard_screen": {},
        "uncertainty": "A census of one preselected shard, with incomplete histories and unvalidated IP linkage; no population CI or full-release extrapolation is valid."
    }
    for n in [20, 50, 100]:
        key = f"{n}_conversations_5_dates"
        result["archive_thresholds"][key] = sum(p["contents"] >= n and p["days"] >= 5 for p in profiles)
        result["archive_thresholds_after_within_shard_screen"][key] = sum(p["contents"] >= n and p["days"] >= 5 and p["screen_pass"] for p in profiles)
    return result


def leakage_components(rows, prefix_characters):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, r in enumerate(rows):
        prefix = " ".join(r["first_user"].split())[:prefix_characters]
        for kind, value in [("ip", r.get("hashed_ip")), ("prompt", prefix)]:
            if not value:
                continue
            token = kind, value
            if token in seen:
                parent[find(i)] = find(seen[token])
            else:
                seen[token] = i
    return [find(i) for i in range(len(rows))]


def split_component(identifier, seed, fraction):
    value = int(sha256(f"split:{seed}:{identifier}".encode()).hexdigest()[:16], 16) / 2**64
    return value < fraction


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def nearest_centroid(train, labels, evaluation):
    labels = np.asarray(labels)
    classes = np.array(sorted(set(labels)))
    centroids = normalize(np.vstack([train[labels == c].mean(axis=0) for c in classes]))
    return classes[np.argmax(normalize(evaluation) @ centroids.T, axis=1)]


def paired_uncertainty(first, second, seed, bootstrap_replicates, permutation_replicates):
    delta = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    if not len(delta):
        raise ValueError("Need nonempty paired observations")
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(bootstrap_replicates)])
    observed = float(delta.mean())
    null = np.array([(delta * rng.choice([-1, 1], len(delta))).mean() for _ in range(permutation_replicates)])
    return {"n": len(delta), "difference": observed, "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "p_sign_flip": float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (1 + len(null)))}


def role_chunks(text, tokenizer, chunk_tokens, max_chunks):
    tokenizer.no_truncation()
    tokenizer.no_padding()
    remaining, chunks = text, []
    while len(chunks) < max_chunks:
        encoded = tokenizer.encode(remaining, add_special_tokens=False)
        if not encoded.ids:
            return chunks, False
        if len(encoded.ids) <= chunk_tokens:
            chunks.append(remaining)
            return chunks, False
        end = encoded.offsets[chunk_tokens][0]
        # A new chunk can begin inside a WordPiece; validate its exact text span.
        while end > 0 and len(tokenizer.encode(remaining[:end], add_special_tokens=False).ids) > chunk_tokens:
            end -= 1
        if end <= 0:
            raise ValueError("Cannot advance within the token budget")
        chunks.append(remaining[:end])
        remaining = remaining[end:]
    return chunks, bool(tokenizer.encode(remaining, add_special_tokens=False).ids)


class Encoder:
    def __init__(self, directory):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / "onnx/model.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.identity = {name: sha256((directory / name).read_bytes()).hexdigest()
                         for name in ["tokenizer.json", "onnx/model.onnx"]}

    def encode(self, chunks):
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=256)
        output = []
        for start in range(0, len(chunks), 32):
            tokens = self.tokenizer.encode_batch(chunks[start:start + 32])
            inputs = {"input_ids": np.array([t.ids for t in tokens], dtype=np.int64),
                      "attention_mask": np.array([t.attention_mask for t in tokens], dtype=np.int64),
                      "token_type_ids": np.array([t.type_ids for t in tokens], dtype=np.int64)}
            inputs = {p.name: inputs[p.name] for p in self.session.get_inputs()}
            hidden = self.session.run(None, inputs)[0]
            mask = inputs["attention_mask"][:, :, None]
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
            output.append(normalize(pooled).astype(np.float32))
        return np.vstack(output)


def main():
    start = time.perf_counter()
    config = json.loads((BASE / "config.json").read_text())
    rows = pq.read_table(BASE / "cache/domain.parquet", columns=list(KEY) + ["conversation_domain"]).to_pylist()
    intent_rows = pq.read_table(BASE / "cache/intent.parquet", columns=list(KEY) + ["user_intent"]).to_pylist()
    domain, domain_stats = unique_label_index(rows, "conversation_domain")
    intent, intent_stats = unique_label_index(intent_rows, "user_intent")
    indexes = {"conversation_domain": domain, "user_intent": intent}
    result = {"eligibility": eligibility(rows), "annotation_keys": {"domain": domain_stats, "intent": intent_stats}}
    sample = pq.read_table(BASE / "cache/content-sample.parquet").to_pylist()
    sample = join_labels(sample, indexes)
    result["content_sample"] = {"physical_rows": len(sample), "label_matched_rows": sum(all(r[k] is not None for k in indexes) for r in sample),
                                "date_min": min(day(r["timestamp"]) for r in sample), "date_max": max(day(r["timestamp"]) for r in sample)}
    # Excluded rows can still bridge eligible rows through an IP or prompt.
    # Build components over the entire acquired universe before any filtering.
    sample = [{**r, "record_id": fingerprint(row_key(r)),
               "first_user": next((m["content"] for m in r["conversation"]
                                   if m["role"] == "user" and (m.get("content") or "").strip()), "")}
              for r in sample]
    source_groups = leakage_components(sample, config["prompt_prefix_characters"])
    source_members = defaultdict(list)
    for group, row in zip(source_groups, sample):
        source_members[group].append(row)
    for group in source_members.values():
        component_id = fingerprint(sorted(r["record_id"] for r in group))
        for row in group:
            row["component_id"] = component_id
    candidates, seen = [], set()
    for r in sample:
        if r["language"] != config["language"] or day(r["timestamp"]) >= config["cutoff"]:
            continue
        if not all(r[k] is not None for k in indexes) or r["conversation_hash"] in seen:
            continue
        role_text = {role: "\n\n".join(m["content"] for m in r["conversation"] if m["role"] == role and m.get("content"))
                     for role in ["user", "assistant"]}
        if not all(t.strip() for t in role_text.values()):
            continue
        seen.add(r["conversation_hash"])
        candidates.append({**r, **role_text})
    members = defaultdict(list)
    for row in candidates:
        members[row["component_id"]].append(row)
    chosen = []
    for group in members.values():
        row = min(group, key=lambda r: r["record_id"])
        chosen.append(row)
    chosen = sorted(chosen, key=lambda r: fingerprint([config["seed"], r["record_id"]]))[:config["conversation_cap"]]
    assert len({r["component_id"] for r in chosen}) == len(chosen)
    result["selection"] = {"source_universe_rows": len(sample), "source_universe_components": len(source_members),
                           "source_component_size_max": max(map(len, source_members.values()), default=0),
                           "eligible_distinct_content_rows": len(candidates), "leakage_components": len(members),
                           "component_size_max": max(map(len, members.values()), default=0), "selected_rows": len(chosen),
                           "rows_in_multi_record_components": sum(len(g) for g in members.values() if len(g) > 1)}
    if len(chosen) < config["minimum_conversations"]:
        result["representation_status"] = "stopped: fewer than the predeclared 200 independent components"
    else:
        manifest = [{"record_id": r["record_id"], "component_id": r["component_id"],
                     "split": "development" if split_component(r["component_id"], config["seed"], config["development_fraction"]) else "evaluation"}
                    for r in chosen]
        (BASE / "split-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        train = np.array([r["split"] == "development" for r in manifest])
        result["selection"]["development_rows"] = int(train.sum())
        result["selection"]["evaluation_rows"] = int((~train).sum())
        encoder = Encoder(ROOT / config["encoder_directory"])
        embeddings, caps = {}, {}
        for role in ["user", "assistant"]:
            blocks, offsets, capped = [], [], 0
            for r in chosen:
                chunks, was_capped = role_chunks(r[role], encoder.tokenizer, config["chunk_tokens"], config["max_role_chunks"])
                offsets.append((len(blocks), len(blocks) + len(chunks)))
                blocks.extend(chunks)
                capped += was_capped
            encoded = encoder.encode(blocks)
            embeddings[role] = normalize(np.vstack([encoded[a:b].mean(axis=0) for a, b in offsets]))
            caps[role] = {"capped_conversations": capped, "chunks": len(blocks)}
        embeddings["equal_role_mean"] = normalize((embeddings["user"] + embeddings["assistant"]) / 2)
        assert all(np.isfinite(e).all() for e in embeddings.values())
        np.savez_compressed(BASE / "cache/embeddings.npz", **embeddings)
        result["encoder"] = {"sha256": encoder.identity, "caps": caps, "threads": 4}
        result["representation_status"] = "completed"
        result["representation"] = {}
        for label in indexes:
            y = np.array([r[label] for r in chosen])
            known = np.array([v in set(y[train]) for v in y[~train]])
            truth = y[~train][known]
            majority = Counter(y[train]).most_common(1)[0][0]
            predictions, accuracies = {}, {}
            measures = {"evaluation_rows": len(truth), "unseen_label_rows": int((~known).sum()),
                        "development_classes": len(set(y[train])), "evaluation_classes": len(set(truth)),
                        "majority_label": majority, "majority_accuracy": float(np.mean(truth == majority)),
                        "development_label_counts": dict(Counter(y[train])), "evaluation_label_counts": dict(Counter(y[~train])), "models": {}}
            for role, e in embeddings.items():
                prediction = nearest_centroid(e[train], y[train], e[~train])[known]
                predictions[role] = prediction
                accuracies[role] = prediction == truth
                measures["models"][role] = {"accuracy": float(np.mean(prediction == truth)),
                                           "macro_recall": float(np.mean([np.mean(prediction[truth == c] == c) for c in set(truth)]))}
            measures["assistant_minus_user"] = paired_uncertainty(accuracies["assistant"], accuracies["user"], config["seed"], config["bootstrap_replicates"], config["permutation_replicates"])
            measures["mixed_minus_user"] = paired_uncertainty(accuracies["equal_role_mean"], accuracies["user"], config["seed"], config["bootstrap_replicates"], config["permutation_replicates"])
            result["representation"][label] = measures
    result["runtime"] = {"seconds": time.perf_counter() - start, "python": platform.python_version(),
                         "platform": platform.platform(), "packages": {k: version(k) for k in ["numpy", "pyarrow", "onnxruntime", "tokenizers", "pytest", "coverage"]}}
    result["provenance"] = {p.name: sha256(p.read_bytes()).hexdigest() for p in [BASE / "config.json", BASE / "PROTOCOL.md", BASE / "audit.py", BASE / "acquire.py"]}
    (BASE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"eligibility":result["eligibility"], "selection":result["selection"], "status":result["representation_status"], "runtime":result["runtime"]}, indent=2))


if __name__ == "__main__":
    main()

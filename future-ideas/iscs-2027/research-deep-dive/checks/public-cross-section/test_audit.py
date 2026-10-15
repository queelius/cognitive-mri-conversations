from datetime import datetime, timezone, timedelta
import numpy as np
import pytest

from audit import (day, unique_label_index, join_labels, leakage_components,
                   normalize, nearest_centroid, paired_uncertainty, split_component,
                   role_chunks, eligibility)


def record(h="a", ip="p", timestamp=None, **kw):
    return dict(conversation_hash=h, hashed_ip=ip, model="m", timestamp=timestamp or datetime(2023, 1, 1),
                turn=1, language="English", state="s", country="c", **kw)


def test_timestamp_uses_utc_day():
    assert day(datetime(2023, 1, 2, 1, tzinfo=timezone(timedelta(hours=4)))) == "2023-01-01"
    assert day(datetime(2023, 1, 2)) == "2023-01-02"


def test_identical_duplicate_labels_do_not_expand_join():
    rows = [record(label="X"), record(label="X")]
    idx, stats = unique_label_index(rows, "label")
    assert stats["duplicate_keys"] == 1
    merged = join_labels([record()], {"label": idx})
    assert len(merged) == 1 and merged[0]["label"] == "X"


def test_conflicting_duplicate_labels_are_rejected():
    with pytest.raises(ValueError, match="Conflicting"):
        unique_label_index([record(label="X"), record(label="Y")], "label")


def test_serialized_none_label_is_missing():
    idx, _ = unique_label_index([record(label="None")], "label")
    assert list(idx.values()) == [None]


def test_join_preserves_unmatched_rows_without_guessing():
    merged = join_labels([record("a"), record("b")], {"label": {}})
    assert len(merged) == 2 and all(r["label"] is None for r in merged)


def test_components_link_ip_and_templates_transitively():
    rows = [dict(hashed_ip="p", first_user="alpha"), dict(hashed_ip="p", first_user="beta"),
            dict(hashed_ip="q", first_user=" beta  "), dict(hashed_ip="r", first_user="gamma")]
    groups = leakage_components(rows, 200)
    assert groups[0] == groups[1] == groups[2] and groups[3] != groups[0]


def test_missing_ips_do_not_merge_everyone():
    rows = [dict(hashed_ip=None, first_user="alpha"), dict(hashed_ip=None, first_user="beta")]
    assert len(set(leakage_components(rows, 200))) == 2


@pytest.mark.parametrize("exclusion", ["language", "missing_label", "cutoff", "missing_assistant"])
def test_selection_retains_bridges_through_ineligible_source_rows(tmp_path, monkeypatch, exclusion):
    """Filtering any bridge before grouping must not create two selected components."""
    import json
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    import audit

    original = audit.BASE
    (tmp_path / "cache").mkdir()
    for name in ["config.json", "PROTOCOL.md", "audit.py", "acquire.py"]:
        shutil.copyfile(original / name, tmp_path / name)
    rows = [record("a", "p"), record("b", "p"), record("c", "q")]
    for row, prompt in zip(rows, ["x", "y", "y"]):
        row.update(conversation=[{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": "response"}],
                   conversation_domain="domain", user_intent="purpose")
    if exclusion == "language":
        rows[1]["language"] = "French"
    elif exclusion == "missing_label":
        rows[1]["conversation_domain"] = None
    elif exclusion == "cutoff":
        rows[1]["timestamp"] = datetime(2025, 1, 1)
    else:
        rows[1]["conversation"] = rows[1]["conversation"][:1]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, tmp_path / "cache/content-sample.parquet")
    for kind, field in [("domain", "conversation_domain"), ("intent", "user_intent")]:
        pq.write_table(table.select(list(audit.KEY) + [field]), tmp_path / f"cache/{kind}.parquet")
    monkeypatch.setattr(audit, "BASE", tmp_path)
    audit.main()
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["selection"]["eligible_distinct_content_rows"] == 2
    assert result["selection"]["selected_rows"] == 1
    assert result["representation_status"].startswith("stopped:")


def test_split_is_stable_and_group_disjoint():
    groups = [str(i) for i in range(100)]
    dev = {g for g in groups if split_component(g, 20261015, .6)}
    evaluation = set(groups) - dev
    assert dev and evaluation and not dev.intersection(evaluation)
    assert all(split_component(g, 20261015, .6) for g in dev)


def test_centroid_classifier_does_not_use_evaluation_labels():
    x = normalize(np.array([[1., 0.], [2., 0.], [0., 1.]]))
    prediction = nearest_centroid(x, ["A", "A", "B"], np.array([[.9, .1], [.1, .9]]))
    assert prediction.tolist() == ["A", "B"]
    assert np.isfinite(normalize(np.zeros((2, 3)))).all()


def test_paired_uncertainty_preserves_pairing():
    zero = paired_uncertainty(np.ones(20), np.ones(20), 2, 100, 100)
    assert zero["difference"] == 0 and zero["ci95"] == [0., 0.] and zero["p_sign_flip"] == 1
    win = paired_uncertainty(np.ones(20), np.zeros(20), 2, 100, 1000)
    assert win["difference"] == 1 and win["ci95"] == [1., 1.] and win["p_sign_flip"] < .01


def test_token_chunk_cap():
    from pathlib import Path
    from tokenizers import Tokenizer
    root = Path(__file__).resolve().parents[5]
    tokenizer = Tokenizer.from_file(str(root / "papers/selective-forgetting/cache/model/tokenizer.json"))
    chunks, capped = role_chunks("word " * 1200, tokenizer, 254, 4)
    assert len(chunks) == 4 and capped
    assert all(len(tokenizer.encode(c, add_special_tokens=False).ids) <= 254 for c in chunks)
    assert role_chunks("", tokenizer, 254, 4) == ([], False)


def test_wordpiece_boundary_does_not_exceed_chunk_budget():
    from pathlib import Path
    from tokenizers import Tokenizer
    root = Path(__file__).resolve().parents[5]
    tokenizer = Tokenizer.from_file(str(root / "papers/selective-forgetting/cache/model/tokenizer.json"))
    text = "uncharacteristically antidisestablishmentarianism. " * 200
    chunks, capped = role_chunks(text, tokenizer, 254, 4)
    assert capped and len(chunks) == 4
    assert all(len(tokenizer.encode(c, add_special_tokens=False).ids) <= 254 for c in chunks)
    assert text.startswith("".join(chunks))


def test_eligibility_counts_unique_content_and_active_dates():
    rows = [record(str(i), timestamp=datetime(2023, 1, 1) + timedelta(days=i % 5)) for i in range(50)]
    rows += [rows[0]]
    result = eligibility(rows)
    assert result["physical_rows"] == 51 and result["unique_content_hashes"] == 50
    assert result["archive_thresholds"]["50_conversations_5_dates"] == 1
    assert result["archive_thresholds"]["100_conversations_5_dates"] == 0

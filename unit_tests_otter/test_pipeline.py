"""Unit tests for the Otter pipeline's own logic, on synthetic data.

No model, no images, no filesystem beyond a tmp dir -- these cover the parts
that are easy to get subtly wrong and expensive to discover in a 24h run:
resampling semantics, the packing invariants, and source partitioning.
"""

import json
import os

import pytest

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.data_otter.mixture import (
    PackingSpec, build_mixture, load_mixture_config, partition_records,
    resample_indices,
)
from llava.data_otter.packing import pack_records, validate_packed_record


def _qa(idx, image=None, n_pairs=1):
    convs = []
    for k in range(n_pairs):
        q = f"question {idx}.{k}"
        if image and k == 0:
            q = f"{DEFAULT_IMAGE_TOKEN}\n{q}"
        convs.append({"from": "human", "value": q})
        convs.append({"from": "gpt", "value": f"answer {idx}.{k}"})
    rec = {"id": f"s{idx}", "conversations": convs}
    if image:
        rec["image"] = image
    return rec


# ---------------------------------------------------------------- resampling
def test_resample_noop():
    idx = list(range(10))
    assert resample_indices(idx, -1, 0) == idx
    assert resample_indices(idx, 0, 0) == idx
    assert resample_indices(idx, 10, 0) == idx


def test_resample_upsample_is_whole_repetitions():
    """2x must be every sample exactly twice, not a lumpy bootstrap."""
    idx = list(range(10))
    out = resample_indices(idx, 20, seed=0)
    assert len(out) == 20
    assert sorted(out) == sorted(idx * 2)


def test_resample_downsample_is_a_subset_without_replacement():
    idx = list(range(100))
    out = resample_indices(idx, 30, seed=0)
    assert len(out) == 30
    assert len(set(out)) == 30
    assert set(out) <= set(idx)


def test_resample_is_seeded():
    idx = list(range(100))
    assert resample_indices(idx, 30, seed=0) == resample_indices(idx, 30, seed=0)


# --------------------------------------------------------------- partitioning
def test_partition_by_image_prefix():
    recs = [
        _qa(0, "coco/train2017/a.jpg"),
        _qa(1, "gqa/images/b.jpg"),
        _qa(2, "coco/train2017/c.jpg"),
        _qa(3),                                    # no image -> text_only
    ]
    buckets = partition_records(recs, "image_prefix")
    assert buckets["coco"] == [0, 2]
    assert buckets["gqa"] == [1]
    assert buckets["text_only"] == [3]


# ------------------------------------------------------------------- packing
def test_pack_merges_shared_image_and_keeps_one_image_token():
    recs = [_qa(i, "gqa/images/same.jpg") for i in range(3)]
    spec = PackingSpec(enabled=True, sources=["gqa"], max_turns=6, max_chars=10000)
    out = pack_records(recs, spec, "gqa", log=lambda *a: None)

    assert len(out) == 1, "three 1-pair records should merge into one 3-pair record"
    packed = out[0]
    assert packed["n_packed"] == 3
    assert len(packed["conversations"]) == 6
    assert validate_packed_record(packed) is None

    # THE invariant: one image -> exactly one <image>, in the first turn.
    n_tokens = sum(t["value"].count(DEFAULT_IMAGE_TOKEN) for t in packed["conversations"])
    assert n_tokens == 1
    assert DEFAULT_IMAGE_TOKEN in packed["conversations"][0]["value"]


def test_pack_respects_max_turns():
    recs = [_qa(i, "gqa/images/same.jpg") for i in range(5)]
    spec = PackingSpec(enabled=True, sources=["gqa"], max_turns=4, max_chars=10000)
    out = pack_records(recs, spec, "gqa", log=lambda *a: None)
    for rec in out:
        assert len(rec["conversations"]) <= 4
        assert validate_packed_record(rec) is None


def test_pack_respects_max_chars():
    recs = [_qa(i, "gqa/images/same.jpg") for i in range(4)]
    spec = PackingSpec(enabled=True, sources=["gqa"], max_turns=100, max_chars=1)
    out = pack_records(recs, spec, "gqa", log=lambda *a: None)
    assert len(out) == 4, "a 1-char budget must prevent any merging"


def test_pack_leaves_distinct_images_alone():
    recs = [_qa(i, f"gqa/images/{i}.jpg") for i in range(3)]
    spec = PackingSpec(enabled=True, sources=["gqa"], max_turns=6, max_chars=10000)
    out = pack_records(recs, spec, "gqa", log=lambda *a: None)
    assert len(out) == 3


def test_pack_skips_malformed_role_ordering():
    bad = {"id": "bad", "image": "gqa/images/x.jpg",
           "conversations": [{"from": "gpt", "value": "answer first"}]}
    good = _qa(1, "gqa/images/x.jpg")
    spec = PackingSpec(enabled=True, sources=["gqa"], max_turns=6, max_chars=10000)
    out = pack_records([bad, good], spec, "gqa", log=lambda *a: None)
    assert bad in out, "a record preprocess() would reject must pass through untouched"


def test_validate_catches_two_image_tokens():
    rec = _qa(0, "gqa/images/x.jpg")
    rec["conversations"][0]["value"] += f" {DEFAULT_IMAGE_TOKEN}"
    assert "2" in (validate_packed_record(rec) or "")


# ------------------------------------------------------- end-to-end mixture
@pytest.fixture
def mixture_dir(tmp_path):
    recs = ([_qa(i, f"coco/{i}.jpg") for i in range(10)]
            + [_qa(100 + i, "gqa/shared.jpg") for i in range(6)]
            + [_qa(200 + i) for i in range(4)])
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(recs))
    return tmp_path, data_path


def _write_cfg(tmp_path, data_path, body):
    cfg = tmp_path / "mix.yaml"
    cfg.write_text(f"""
version: 1
defaults: {{image_folder: {tmp_path}}}
partition: {{data_path: {data_path}, by: image_prefix}}
{body}
seed: 0
""")
    return str(cfg)


def test_build_mixture_applies_resampling(mixture_dir):
    tmp_path, data_path = mixture_dir
    cfg_path = _write_cfg(tmp_path, data_path, """
sources:
  coco: {num_samples: 20}
  gqa: {num_samples: -1}
  text_only: {num_samples: -1}
packing: {enabled: false}
""")
    cfg = load_mixture_config(cfg_path)
    records, sources, folders = build_mixture(cfg, log=lambda *a: None)
    assert sources.count("coco") == 20      # 10 upsampled 2x
    assert sources.count("gqa") == 6
    assert sources.count("text_only") == 4


def test_packing_runs_before_resampling(mixture_dir):
    """The ordering that stops an upsample being silently undone.

    gqa's 6 records share one image and pack into 2 (max_turns=6 -> 3 each).
    Asking for 8 samples must then upsample those 2 packs to 8, NOT re-merge
    duplicates back down.
    """
    tmp_path, data_path = mixture_dir
    cfg_path = _write_cfg(tmp_path, data_path, """
sources:
  coco: {num_samples: -1}
  gqa: {num_samples: 8}
  text_only: {num_samples: -1}
packing: {enabled: true, sources: [gqa], max_turns: 6, max_chars: 100000}
""")
    cfg = load_mixture_config(cfg_path)
    records, sources, _ = build_mixture(cfg, log=lambda *a: None)
    assert sources.count("gqa") == 8
    gqa = [r for r, s in zip(records, sources) if s == "gqa"]
    assert all(r.get("n_packed") == 3 for r in gqa)
    assert all(validate_packed_record(r) is None for r in gqa)


def test_unknown_source_is_an_error(mixture_dir):
    tmp_path, data_path = mixture_dir
    cfg_path = _write_cfg(tmp_path, data_path, """
sources:
  cocoo: {num_samples: -1}
packing: {enabled: false}
""")
    cfg = load_mixture_config(cfg_path)
    with pytest.raises(ValueError, match="cocoo"):
        build_mixture(cfg, log=lambda *a: None)


def test_packing_source_must_be_declared(mixture_dir):
    tmp_path, data_path = mixture_dir
    cfg_path = _write_cfg(tmp_path, data_path, """
sources:
  coco: {num_samples: -1}
packing: {enabled: true, sources: [nope]}
""")
    with pytest.raises(ValueError, match="nope"):
        load_mixture_config(cfg_path)


def test_signature_changes_with_the_mixture(mixture_dir):
    tmp_path, data_path = mixture_dir
    a = load_mixture_config(_write_cfg(tmp_path, data_path,
                                       "sources: {coco: {num_samples: -1}}\npacking: {enabled: false}"))
    sig_a = a.signature()
    b = load_mixture_config(_write_cfg(tmp_path, data_path,
                                       "sources: {coco: {num_samples: 5}}\npacking: {enabled: false}"))
    assert sig_a != b.signature(), "cache keys must not collide across different mixtures"


# ----------------------------------------------------------------- telemetry
def test_telemetry_gap_and_homogeneity():
    from llava.data_otter.telemetry import OtterTelemetry

    tel = OtterTelemetry(log_every=1)
    # Homogeneous ocr_vqa batch with a real budget gap.
    tel.record({"sources": ["ocr_vqa", "ocr_vqa"], "batch_size": 2, "supervised_tokens": 40,
                "padded_tokens": 4, "seq_len": 300, "n_packed": [3, 3], "image_fallbacks": 0},
               {"loss/ce_tok256": 1.0, "loss/ce_tok16": 1.5}, total_loss=1.25)
    # Mixed batch: counts toward the level means, not the per-source ones.
    tel.record({"sources": ["coco", "gqa"], "batch_size": 2, "supervised_tokens": 20,
                "padded_tokens": 2, "seq_len": 300, "n_packed": [1, 1], "image_fallbacks": 0},
               {"loss/ce_tok256": 1.0, "loss/ce_tok16": 1.0}, total_loss=1.0)

    out = tel.flush(1)
    assert out["otter/gap/ocr_vqa"] == pytest.approx(0.5)
    assert out["otter/gap_n/ocr_vqa"] == 1.0
    assert out["otter/hom_frac"] == pytest.approx(0.5)
    # supervised is tracked PER SAMPLE (40/2=20, then 20/2=10, each weighted by
    # the 2 samples in its batch -> mean 15) against a per-batch seq_len of 300.
    assert out["otter/supervised_tokens"] == pytest.approx(15.0)
    assert out["otter/supervised_frac"] == pytest.approx(15 / 300)
    assert "otter/loss/coco" not in out, "mixed batches must not be attributed to a source"
    assert tel.n_batches == 0, "flush must reset the window"


# ------------------------------------------------------- source-grouped sampler
def test_source_grouped_sampler_is_homogeneous_and_complete():
    """Every batch single-source, and no sample lost or duplicated."""
    from llava.data_otter.sampler import SourceGroupedLengthSampler, homogeneity_of

    n_per = 40
    sources, lengths = [], []
    for s in ["coco", "gqa", "ocr_vqa", "textvqa", "vg"]:
        for k in range(n_per):
            sources.append(s)
            # Overlapping length ranges across sources: this is exactly the
            # condition that defeats the stock length-grouped sampler.
            lengths.append(100 + (k * 7) % 300)

    sampler = SourceGroupedLengthSampler(
        batch_size=2, world_size=8, lengths=lengths, source_names=sources, seed=0)
    order = list(sampler)

    assert sorted(order) == list(range(len(sources))), "sampler must be a permutation"
    assert homogeneity_of(order, sources, batch_size=2) == 1.0


def test_source_grouped_sampler_handles_odd_source_sizes():
    """A source with an odd count must not make a cross-source batch mid-stream."""
    from llava.data_otter.sampler import SourceGroupedLengthSampler

    sources = ["a"] * 5 + ["b"] * 7 + ["c"] * 3
    lengths = [100 + i for i in range(len(sources))]
    sampler = SourceGroupedLengthSampler(
        batch_size=2, world_size=4, lengths=lengths, source_names=sources, seed=0)
    order = list(sampler)
    assert sorted(order) == list(range(len(sources)))

    # 5//2 + 7//2 + 3//2 = 2+3+1 = 6 full single-source batches, then the tail.
    n_full = 6
    for i in range(n_full):
        chunk = order[i * 2:(i + 1) * 2]
        assert len({sources[j] for j in chunk}) == 1, f"batch {i} is mixed"


def test_source_grouped_sampler_is_deterministic_and_mixes_sources():
    from llava.data_otter.sampler import SourceGroupedLengthSampler

    sources = (["coco"] * 60) + (["ocr_vqa"] * 60)
    lengths = [100 + (i * 3) % 200 for i in range(len(sources))]
    mk = lambda: SourceGroupedLengthSampler(
        batch_size=2, world_size=4, lengths=lengths, source_names=sources, seed=7)
    assert list(mk()) == list(mk()), "same seed must give the same order"

    # Batches must be interleaved, not all of one source then all of the other:
    # an accumulation window has to average over a real mixture.
    order = list(mk())
    batch_sources = [sources[order[i]] for i in range(0, len(order), 2)]
    first_half = set(batch_sources[:len(batch_sources) // 2])
    assert len(first_half) == 2, "sources must be interleaved across the epoch"

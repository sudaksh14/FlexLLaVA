"""Token-accurate sequence lengths for the length-grouped sampler.

WHY THIS EXISTS
---------------
LazySupervisedDataset.lengths estimates a sample's length as

    sum(len(conv['value'].split()) for conv in sample['conversations']) + 128

i.e. whitespace WORDS plus a hard-coded 128 visual tokens.  Both terms are
wrong in ways that cost throughput:

  * words != tokens.  The ratio is ~1.3 for prose and far higher for the OCR
    and coordinate-heavy strings in ocr_vqa / textvqa / vg, so the sampler's
    idea of "same length" does not match what actually lands in the batch.
  * the visual prefix is 128 in that formula but `tok_levels[0]` (256 by
    default) in this project, and the elastic forward pays it once per active
    level.

LengthGroupedSampler sorts by these numbers to fill batches with
similar-length sequences; when the numbers are wrong the batches carry more
padding, and every padded token is a real forward/backward on every level.

Computing true lengths means tokenising the corpus once (~665k samples).  That
is minutes of CPU, not seconds, so it is cached to disk and keyed on everything
that could change the answer.  If no cache exists and building is disabled, the
dataset falls back to the old heuristic and says so -- a missing cache degrades
throughput, it never changes correctness.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

# Tokens each turn costs beyond its own text: role marker, separator, newline.
# Measured range across the templates in llava/conversation.py is 3-6 (vicuna_v1
# "USER: "/" ASSISTANT: " ~4, chatml "<|im_start|>user\n"/"<|im_end|>\n" ~5,
# phi3 "<|user|>"/"<|end|>" ~2).  This is a sorting key, not a budget, so a
# small constant bias that applies to every sample equally is harmless.
PER_TURN_OVERHEAD_TOKENS = 5


def cache_key(mixture_signature: str,
              tokenizer_name: str,
              visual_tokens: int,
              model_max_length: int) -> str:
    raw = f"{mixture_signature}|{tokenizer_name}|{visual_tokens}|{model_max_length}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"otter_lengths_{key}.json")


def heuristic_lengths(records: List[Dict[str, Any]], visual_tokens: int) -> List[int]:
    """The shipped word-count estimate, with the correct visual budget.

    Used when no cache is available.  Kept deliberately close to
    LazySupervisedDataset.lengths so a run without a cache behaves like the
    current pipeline rather than like some third thing.
    """
    out = []
    for rec in records:
        n = sum(len(t.get("value", "").split()) for t in rec.get("conversations", []))
        out.append(n + (visual_tokens if rec.get("image") else 0))
    return out


def compute_lengths(records: List[Dict[str, Any]],
                    tokenizer,
                    visual_tokens: int,
                    batch_size: int = 1000,
                    log=print) -> List[int]:
    """True token lengths, by tokenising every turn.

    Turns are tokenised in batches with the fast path (no special tokens, no
    padding, no truncation) -- we want the raw content length, since truncation
    is exactly what we are trying to predict and avoid.
    """
    lengths: List[int] = []
    # Flatten to one list of strings so the tokenizer sees large batches; then
    # fold the per-turn counts back onto their records.
    flat: List[str] = []
    spans: List[tuple] = []
    for rec in records:
        turns = rec.get("conversations", [])
        start = len(flat)
        flat.extend(t.get("value", "") for t in turns)
        spans.append((start, len(flat), bool(rec.get("image")), len(turns)))

    counts: List[int] = []
    total = len(flat)
    for i in range(0, total, batch_size):
        chunk = flat[i:i + batch_size]
        enc = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        counts.extend(len(ids) for ids in enc)
        if total and (i // batch_size) % 100 == 0:
            log(f"[otter-len] tokenised {min(i + batch_size, total)}/{total} turns")

    for start, end, has_image, n_turns in spans:
        n = sum(counts[start:end]) + n_turns * PER_TURN_OVERHEAD_TOKENS
        if has_image:
            n += visual_tokens
        lengths.append(n)
    return lengths


def load_or_build(records: List[Dict[str, Any]],
                  tokenizer,
                  visual_tokens: int,
                  mixture_signature: str,
                  cache_dir: Optional[str],
                  build_if_missing: bool = False,
                  log=print) -> tuple:
    """Return (lengths, provenance) where provenance is 'cache' | 'built' | 'heuristic'."""
    tok_name = getattr(tokenizer, "name_or_path", "unknown")
    max_len = getattr(tokenizer, "model_max_length", -1)
    key = cache_key(mixture_signature, tok_name, visual_tokens, max_len)

    if cache_dir:
        path = cache_path(cache_dir, key)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    blob = json.load(f)
                if blob.get("n") == len(records):
                    log(f"[otter-len] loaded token lengths from {path}")
                    return blob["lengths"], "cache"
                log(f"[otter-len] cache {path} has {blob.get('n')} entries but the mixture has "
                    f"{len(records)}; ignoring it")
            except (OSError, ValueError, KeyError) as e:
                log(f"[otter-len] could not read cache {path}: {e}")

    if build_if_missing:
        log(f"[otter-len] building token lengths for {len(records)} records "
            f"(visual_tokens={visual_tokens}) -- this is a one-off, then cached")
        lengths = compute_lengths(records, tokenizer, visual_tokens, log=log)
        if cache_dir:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                path = cache_path(cache_dir, key)
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"key": key, "n": len(records), "visual_tokens": visual_tokens,
                               "tokenizer": tok_name, "lengths": lengths}, f)
                os.replace(tmp, path)   # atomic: a killed job never leaves a half-written cache
                log(f"[otter-len] wrote {path}")
            except OSError as e:
                log(f"[otter-len] could not write cache: {e}")
        return lengths, "built"

    log("[otter-len] NOTE: no token-length cache; falling back to the word-count heuristic. "
        "Batches will carry more padding than necessary. Build one with "
        "`python -m llava.data_otter.prepare --build lengths ...`")
    return heuristic_lengths(records, visual_tokens), "heuristic"

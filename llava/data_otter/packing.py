"""Pack QA pairs that share an image into one multi-turn conversation.

WHY THIS EXISTS
---------------
Most of mix665k is a single short QA per image (gqa, ocr_vqa, textvqa in
particular).  Every such sample costs a full visual prefix -- 256 tokens at the
teacher level -- to supervise a handful of answer tokens, and the elastic
trainer pays that prefix once PER ACTIVE LEVEL (teacher + n_sample_students).
Packing the several questions that already exist for the same image into one
conversation amortises the prefix over more supervised tokens, which is the
cheapest available lever on the 24h+ Stage-2 epoch (5197 steps).

It is also the format the multi-turn masking code is best exercised on: Otter
does the same thing via its `rel_ins_ids` in-context grouping
(pipeline/mimicit_utils/mimicit_dataset.py:352-386).

THE CORRECTNESS CONSTRAINT
--------------------------
`prepare_inputs_labels_for_multimodal` splits input_ids on IMAGE_TOKEN_INDEX
and splices in one image feature block per occurrence.  A packed record carries
exactly ONE image, so it must contain exactly ONE `<image>` token -- otherwise
the splice consumes an image that was never passed and the batch either crashes
or, worse, silently mis-aligns the visual prefix.  `_merge_conversations` below
strips `<image>` from every turn it appends and guarantees the single leading
occurrence, and `validate_packed_record` re-checks it.

Packing runs BEFORE resampling (see mixture.build_mixture).  If it ran after,
grouping by image would merge an upsampled source's duplicate copies back into
a single pack and silently undo the upsample.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from llava.constants import DEFAULT_IMAGE_TOKEN

from .mixture import PackingSpec


def _strip_image_tokens(text: str) -> str:
    """Remove every <image> marker and tidy the whitespace it leaves behind."""
    out = text.replace(DEFAULT_IMAGE_TOKEN, "")
    # The canonical form is "<image>\n<question>", so removal usually leaves a
    # leading newline.  Strip only at the edges; interior formatting is content.
    return out.strip()


def _record_chars(record: Dict[str, Any]) -> int:
    return sum(len(turn.get("value", "")) for turn in record.get("conversations", []))


def _merge_conversations(group: List[Dict[str, Any]], has_image: bool) -> List[Dict[str, str]]:
    """Concatenate several records' turns into one conversation.

    The first record keeps its `<image>` marker (normalised to the front of its
    first turn, which is what preprocess_multimodal would do anyway); every
    later record has its markers stripped.
    """
    merged: List[Dict[str, str]] = []
    for k, rec in enumerate(group):
        for j, turn in enumerate(rec.get("conversations", [])):
            value = _strip_image_tokens(turn.get("value", ""))
            if has_image and k == 0 and j == 0:
                value = f"{DEFAULT_IMAGE_TOKEN}\n{value}" if value else DEFAULT_IMAGE_TOKEN
            merged.append({"from": turn["from"], "value": value})
    return merged


def validate_packed_record(record: Dict[str, Any]) -> Optional[str]:
    """Return a human-readable problem with this record, or None if it is fine.

    Used both by the packer (as an assertion on its own output) and by the
    prerun verification gate, so a packing bug cannot reach a 24h training run.
    """
    convs = record.get("conversations") or []
    if not convs:
        return "no conversations"
    n_image_tokens = sum(t.get("value", "").count(DEFAULT_IMAGE_TOKEN) for t in convs)
    if record.get("image"):
        if n_image_tokens != 1:
            return (f"record has an image but {n_image_tokens} '{DEFAULT_IMAGE_TOKEN}' tokens "
                    f"(must be exactly 1)")
        if DEFAULT_IMAGE_TOKEN not in convs[0].get("value", ""):
            return f"'{DEFAULT_IMAGE_TOKEN}' is not in the first turn"
    elif n_image_tokens:
        return f"record has no image but {n_image_tokens} '{DEFAULT_IMAGE_TOKEN}' tokens"
    if len(convs) % 2 != 0:
        return f"odd number of turns ({len(convs)}); expected human/gpt pairs"
    for j, turn in enumerate(convs):
        expected = "human" if j % 2 == 0 else "gpt"
        if turn.get("from") != expected:
            return f"turn {j} is from {turn.get('from')!r}, expected {expected!r}"
    return None


def pack_records(records: List[Dict[str, Any]],
                 spec: PackingSpec,
                 source_name: str,
                 log=print) -> List[Dict[str, Any]]:
    """Group `records` by image and merge each group into multi-turn samples.

    Records that cannot be packed (no image, group of one, already at the turn
    budget, malformed role ordering) pass through untouched, so this is safe to
    apply to a whole source without curating it first.
    """
    by_image: Dict[str, List[int]] = defaultdict(list)
    passthrough: List[int] = []

    for i, rec in enumerate(records):
        image = rec.get("image")
        convs = rec.get("conversations") or []
        # Only ever merge well-formed human/gpt-alternating records: the
        # multi-turn masking in preprocess_v1/preprocess_mpt asserts that
        # ordering, and a mismatch there fully IGNORE-masks the sample.
        well_formed = (
            len(convs) > 0
            and len(convs) % 2 == 0
            and all(t.get("from") == ("human" if j % 2 == 0 else "gpt")
                    for j, t in enumerate(convs))
        )
        if image and well_formed and len(convs) < spec.max_turns:
            by_image[image].append(i)
        else:
            passthrough.append(i)

    rng = random.Random(f"{spec.seed}:{source_name}")
    out: List[Dict[str, Any]] = []
    n_packed_records = 0
    n_packs = 0

    for image in sorted(by_image):                    # sorted -> deterministic
        members = by_image[image]
        if len(members) < spec.min_pack:
            out.extend(records[i] for i in members)
            continue
        order = list(members)
        if spec.shuffle_within_pack:
            rng.shuffle(order)

        group: List[Dict[str, Any]] = []
        group_turns = 0
        group_chars = 0

        def flush(group, group_turns):
            nonlocal n_packed_records, n_packs
            if not group:
                return
            if len(group) < spec.min_pack:
                out.extend(group)
                return
            packed = dict(group[0])                   # shallow copy; never mutate the original
            packed["conversations"] = _merge_conversations(group, has_image=True)
            packed["id"] = f"pack:{group[0].get('id', 'na')}+{len(group) - 1}"
            packed["n_packed"] = len(group)
            problem = validate_packed_record(packed)
            if problem is not None:
                # Never train on a record we cannot vouch for; fall back to the
                # unpacked originals, which were already valid on their own.
                log(f"[otter-pack] {source_name}: refusing pack for {image!r} ({problem}); "
                    f"emitting {len(group)} original records instead")
                out.extend(group)
                return
            out.append(packed)
            n_packed_records += len(group)
            n_packs += 1

        for i in order:
            rec = records[i]
            n_turns = len(rec.get("conversations", []))
            n_chars = _record_chars(rec)
            over_turns = group and group_turns + n_turns > spec.max_turns
            over_chars = group and group_chars + n_chars > spec.max_chars
            if over_turns or over_chars:
                flush(group, group_turns)
                group, group_turns, group_chars = [], 0, 0
            group.append(rec)
            group_turns += n_turns
            group_chars += n_chars
        flush(group, group_turns)

    out.extend(records[i] for i in passthrough)

    if n_packs:
        log(f"[otter-pack] {source_name}: {len(records)} records -> {len(out)} "
            f"({n_packed_records} records merged into {n_packs} packs, "
            f"avg {n_packed_records / n_packs:.1f} per pack, "
            f"{len(passthrough)} unpackable passed through)")
    else:
        log(f"[otter-pack] {source_name}: nothing packable ({len(records)} records unchanged)")
    return out

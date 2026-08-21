"""Image-existence prescan, so unreadable samples are dropped instead of swapped.

WHY THIS EXISTS
---------------
LazySupervisedDataset.__getitem__ handles a missing/corrupt image by retrying a
RANDOM other index, up to 10 times (llava/train/train.py:1048-1058).  It keeps
long runs alive, which is why it was written, but it has two costs:

  * it silently reweights the mixture -- every bad sample becomes an extra draw
    of some random good one, and nothing counts how often that happens;
  * it is not reproducible.  Two runs over the same data see different data.

Both matter more once the mixture itself is the thing under test (see
mixture.py): an ablation that upweights ocr_vqa is not measuring what it claims
if unreadable ocr_vqa samples are being replaced by random coco ones.

The prescan resolves this once, offline: stat every referenced image, cache the
list of missing ones, and drop those records at mixture-build time so the run
starts from a known-good, fully-reproducible set.  Runtime failures that slip
through anyway (a file that stats but fails to decode) fall back to the NEXT
index rather than a random one, and are counted for telemetry.

This is the cheap half of the storage fix.  The expensive half -- packing the
~600k small files into parquet/webdataset shards to stop per-sample filesystem
hits on /var/scratch, the way Otter stores base64 in parquet -- is a separate
offline job and is not attempted here.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple


def manifest_key(mixture_signature: str) -> str:
    return hashlib.sha1(mixture_signature.encode("utf-8")).hexdigest()[:16]


def manifest_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"otter_missing_images_{key}.json")


def scan_missing(records: List[Dict[str, Any]],
                 source_names: List[str],
                 image_folders: Dict[str, str],
                 log=print) -> List[str]:
    """Return the sorted list of image paths (relative, as stored) that do not exist.

    Only os.path.exists, not a decode: a full decode of 600k images would take
    hours and the overwhelmingly common failure in this dataset is an absent
    file (ocr_vqa ships with a known handful).
    """
    checked: Set[str] = set()
    missing: Set[str] = set()
    for rec, source in zip(records, source_names):
        image = rec.get("image")
        if not image or image in checked:
            continue
        checked.add(image)
        folder = image_folders.get(source) or ""
        if not os.path.exists(os.path.join(folder, image)):
            missing.add(image)
    log(f"[otter-manifest] checked {len(checked)} distinct images, {len(missing)} missing")
    return sorted(missing)


def load_missing(cache_dir: Optional[str],
                 mixture_signature: str,
                 log=print) -> Optional[Set[str]]:
    """Load a cached missing-image list, or None if there isn't one."""
    if not cache_dir:
        return None
    path = manifest_path(cache_dir, manifest_key(mixture_signature))
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            blob = json.load(f)
        log(f"[otter-manifest] loaded {len(blob['missing'])} known-missing images from {path}")
        return set(blob["missing"])
    except (OSError, ValueError, KeyError) as e:
        log(f"[otter-manifest] could not read {path}: {e}")
        return None


def save_missing(cache_dir: str, mixture_signature: str, missing: List[str], log=print) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = manifest_path(cache_dir, manifest_key(mixture_signature))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"missing": missing, "n_missing": len(missing)}, f)
    os.replace(tmp, path)
    log(f"[otter-manifest] wrote {path}")
    return path


def drop_missing(records: List[Dict[str, Any]],
                 source_names: List[str],
                 missing: Set[str],
                 log=print) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Filter out records whose image is known-missing, reporting per source."""
    if not missing:
        return records, source_names
    keep_r, keep_s = [], []
    dropped: Dict[str, int] = {}
    for rec, source in zip(records, source_names):
        image = rec.get("image")
        if image and image in missing:
            dropped[source] = dropped.get(source, 0) + 1
            continue
        keep_r.append(rec)
        keep_s.append(source)
    if dropped:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
        log(f"[otter-manifest] dropped {sum(dropped.values())} records with missing images "
            f"({detail})")
    return keep_r, keep_s

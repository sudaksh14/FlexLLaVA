"""TextVQA val on the ViCrop ground-truth-box subset, for the crop pilot.

Three input modes over the same questions (see the yaml next to this file):
  full        original image only            (control)
  crop        GT-box square crop only        (single pass, CoInfer-style)
  globalcrop  original image + GT-box crop   (ViCrop's "human-CROP" protocol)

Scoring is the stock lmms-eval TextVQA scorer, unchanged.
"""
import json
import os

from lmms_eval.tasks.textvqa.utils import textvqa_doc_to_text as _doc_to_text
from lmms_eval.tasks.textvqa.utils import textvqa_process_results as _process_results

SIDECAR = os.environ.get("CROPPILOT_SIDECAR", "/var/scratch/skalra/flexllava/croppilot/textvqa_val_gtbox.json")
_SIDE = None


def _side():
    global _SIDE
    if _SIDE is None:
        with open(SIDECAR) as f:
            _SIDE = json.load(f)
    return _SIDE


def process_docs(dataset):
    side = _side()
    return dataset.filter(lambda q: str(q) in side, input_columns=["question_id"])


def gt_square_crop(img, bbox_xywh):
    """Smallest square around the GT box (ViCrop Sec. 3), clipped to the image.

    If the box is wider/taller than the image's short side the crop is clipped
    to the image extent in that dimension (so it is a rectangle, not a square).
    """
    W, H = img.size
    x, y, w, h = bbox_xywh
    s = max(w, h)
    sx, sy = min(s, W), min(s, H)
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = min(max(cx - sx / 2.0, 0.0), W - sx)
    y0 = min(max(cy - sy / 2.0, 0.0), H - sy)
    box = (int(x0), int(y0), int(min(W, round(x0 + sx + 0.5))), int(min(H, round(y0 + sy + 0.5))))
    return img.crop(box)


def _crop(doc):
    img = doc["image"].convert("RGB")
    return gt_square_crop(img, _side()[str(doc["question_id"])]["bbox_xywh"])


def doc_to_visual_full(doc):
    return [doc["image"].convert("RGB")]


def doc_to_visual_crop(doc):
    return [_crop(doc)]


def doc_to_visual_globalcrop(doc):
    return [doc["image"].convert("RGB"), _crop(doc)]


def doc_to_text_globalcrop(doc, model_specific_prompt_kwargs=None):
    # Two image slots, stated explicitly: the stock wrapper only inserts one
    # <image> token per document when the prompt has none.
    return "<image>\n<image>\n" + _doc_to_text(doc, model_specific_prompt_kwargs)


def process_results(doc, result):
    out = _process_results(doc, result)
    return {"exact_match": out["exact_match"]}


def doc_to_text_stock(doc, model_specific_prompt_kwargs=None):
    return _doc_to_text(doc, model_specific_prompt_kwargs)

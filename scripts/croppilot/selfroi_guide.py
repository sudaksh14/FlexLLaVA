"""I-3 stage A: self-guided ROI saliency from a low-budget pass of the nested model.

For each TextVQA question (ViCrop GT-box subset) run the model on the *global*
image at a guide budget (tok_level index; 3 -> 16 tokens, 0 -> 256 tokens) and
record, per LM layer, the answer-to-image saliency on the 24x24 CLIP patch grid:

    s_l(p) = sum_t A_st[l, t] * A_ti[t, p]

A_st  : attention of the last prompt position (the one that predicts the first
        answer token) to the n visual-token positions, head-averaged   (ViCrop)
A_ti  : how much each visual token draws on each of the 576 patches.  Anchors
        are uniform over their pooling window; the learned queries use the
        resampler's cross-attention weights (mean over its layers), captured
        with forward hooks.

Both the question prompt and ViCrop's generic prompt are run, so stage B can
form raw, relative (rel-att) and all-layer (SGL-style) maps.  The guide pass's
own greedy answer, its score and its confidence are stored too (they are the
inputs to an early-exit gate).

Run through sbatch (GPU, node206-208): see selfroi_guide.sh.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/var/scratch/skalra/.cache/huggingface")
sys.path.insert(0, "/home/skalra/FlexLLaVA")
sys.path.insert(0, "/home/skalra/FlexLLaVA/lmms-eval")

import datasets
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from lmms_eval.models.llava_elastic import LlavaElastic
from lmms_eval.tasks.textvqa.utils import textvqa_doc_to_text, textvqa_process_results

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/var/scratch/skalra/flexllava/checkpoints/elastic-finetune-tinyllama-v8-parcel")
ap.add_argument("--level", type=int, default=3, help="tok_level index of the guide pass (3=16 tok, 0=256 tok)")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="/var/scratch/skalra/flexllava/croppilot/selfroi")
ap.add_argument("--sidecar", default="/var/scratch/skalra/flexllava/croppilot/textvqa_val_gtbox.json")
args = ap.parse_args()

GENERIC = "Write a general description of the image."
POST = "\nAnswer the question using a single word or phrase."
CACHE = "/var/scratch/skalra/.cache/huggingface/datasets"

side = json.load(open(args.sidecar))
ds = datasets.load_dataset("lmms-lab/textvqa", split="validation", cache_dir=CACHE)
ds = ds.filter(lambda q: str(q) in side, input_columns=["question_id"])
idx = list(range(args.shard, len(ds), args.nshards))
if args.limit:
    idx = idx[: args.limit]
print(f"guide level idx={args.level}  shard {args.shard}/{args.nshards}  questions={len(idx)}", flush=True)

wrap = LlavaElastic(pretrained=args.ckpt, tok_level=args.level, device_map="cuda:0",
                    conv_template="vicuna_v1", attn_implementation="eager")
model, tok, proc = wrap.model, wrap._tokenizer, wrap._image_processor
model.eval()
dev = next(model.parameters()).device
cfg = model.config
n_levels = model.elastic_engine.cfg.tok_levels
n_tok = int(n_levels[args.level])
print("transformers", __import__("transformers").__version__, "| image_aspect_ratio =",
      getattr(cfg, "image_aspect_ratio", None), "| visual tokens =", n_tok,
      "| attn impl =", getattr(cfg, "_attn_implementation", None), flush=True)

res = model.elastic_resampler
P = 576
n_p = res.n_anchors_for(n_tok, P)
n_q = n_tok - n_p
g = int(round(n_p ** 0.5))
k = int(round(P ** 0.5)) // g  # pooling window side, in patches
# A_ti for anchors: uniform over their k x k window of the 24x24 grid.
anchor_w = torch.zeros(n_p, P)
for a in range(n_p):
    ar, ac = divmod(a, g)
    for r in range(ar * k, (ar + 1) * k):
        for c in range(ac * k, (ac + 1) * k):
            anchor_w[a, r * 24 + c] = 1.0 / (k * k)
anchor_w = anchor_w.to(dev)
print(f"anchors={n_p} (pool {k}x{k})  queries={n_q}", flush=True)

captured = []
hooks = []
for layer in res.layers:
    def _hook(mod, inp, out):
        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
            captured.append(out[1].detach().float())
    hooks.append(layer["cross_attn"].register_forward_hook(_hook))


def build(question_text, img):
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question_text)
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(dev)
    return ids


@torch.no_grad()
def saliency(ids, images, size):
    """-> (L, 576) saliency per LM layer for the last prompt position."""
    j = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
    captured.clear()
    _, pos, am, _, emb, _ = model.prepare_inputs_labels_for_multimodal(
        ids, None, torch.ones_like(ids), None, None, images, image_sizes=[size],
        matryoshka_vis_token_scale=args.level)
    n = emb.shape[1] - (ids.shape[1] - 1)
    assert n == n_tok, (n, n_tok)
    ti_q = torch.stack(captured).mean(0)[0]            # (n_q, 576), mean over resampler layers
    assert ti_q.shape == (n_q, P), ti_q.shape
    A_ti = torch.cat([anchor_w, ti_q], dim=0)          # token order: anchors then queries
    out = model.model(inputs_embeds=emb, attention_mask=am, position_ids=pos,
                      output_attentions=True, use_cache=False, return_dict=True)
    assert out.attentions is not None and out.attentions[0] is not None, "no attentions returned"
    A_st = torch.stack([a[0, :, -1, j:j + n].float().mean(0) for a in out.attentions])   # (L, n)
    return (A_st @ A_ti).cpu().numpy()                  # (L, 576)


@torch.no_grad()
def answer(ids, images, size):
    gen = model.generate(ids, images=images, image_sizes=[size], attention_mask=torch.ones_like(ids),
                         pad_token_id=tok.eos_token_id, do_sample=False, num_beams=1, max_new_tokens=16,
                         use_cache=True, matryoshka_vis_token_scale=args.level,
                         output_scores=True, return_dict_in_generate=True)
    seq = gen.sequences[0]
    lps = []
    for t, sc in enumerate(gen.scores):
        lps.append(torch.log_softmax(sc[0].float(), -1)[seq[t]].item())
    text = tok.decode(seq, skip_special_tokens=True).split("ASSISTANT:")[0].strip()
    return text, lps


qids, raw, gen_sal, p1_text, p1_acc, p1_first, p1_mean = [], [], [], [], [], [], []
t0 = time.time()
for c, i in enumerate(idx):
    r = ds[i]
    img = r["image"].convert("RGB")
    images = process_images([img], proc, cfg).to(dtype=torch.float16, device=dev)
    qtext = textvqa_doc_to_text(r, {"pre_prompt": "", "post_prompt": POST, "ocr": False})
    ids_q = build(qtext, img)
    ids_g = build(GENERIC, img)
    raw.append(saliency(ids_q, images, img.size).astype(np.float16))
    gen_sal.append(saliency(ids_g, images, img.size).astype(np.float16))
    text, lps = answer(ids_q, images, img.size)
    acc = textvqa_process_results({"answers": list(r["answers"]), "question_id": r["question_id"]}, [text])["exact_match"]
    qids.append(int(r["question_id"]))
    p1_text.append(text)
    p1_acc.append(float(acc))
    p1_first.append(float(np.exp(lps[0])) if lps else float("nan"))
    p1_mean.append(float(np.exp(np.mean(lps))) if lps else float("nan"))
    if c % 100 == 0:
        print(f"  {c}/{len(idx)}  {time.time() - t0:.0f}s  running acc {100 * np.mean(p1_acc):.2f}", flush=True)

os.makedirs(args.out, exist_ok=True)
tag = f"guide{n_tok}_shard{args.shard}of{args.nshards}"
np.savez_compressed(f"{args.out}/{tag}.npz", qids=np.array(qids), raw=np.stack(raw), generic=np.stack(gen_sal),
                    p1_acc=np.array(p1_acc), p1_first_prob=np.array(p1_first), p1_mean_prob=np.array(p1_mean))
json.dump({"qids": qids, "text": p1_text}, open(f"{args.out}/{tag}_answers.json", "w"))
print(f"done {tag}: {len(qids)} questions, guide-pass acc {100 * np.mean(p1_acc):.2f}, {time.time() - t0:.0f}s", flush=True)

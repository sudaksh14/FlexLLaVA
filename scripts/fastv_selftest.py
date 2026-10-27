"""Self-test for llava/model/fastv.py (GPU).  python scripts/fastv_selftest.py <dense_ckpt>
 1. ratio=0 reproduces the unpatched eager model token for token;
 2. ratio=0.75 keeps round(576*0.25)=144 visual tokens and changes the output;
 3. batched (left-padded, different prompt lengths) greedy output == one-by-one output for ratio=0.75."""
import sys
import numpy as np
import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.fastv import enable_fastv

path = sys.argv[1]
tok, model, proc, _ = load_pretrained_model(path, None, "tinyllama-fastv-test", device_map="cuda:0", attn_implementation="eager")
model.eval(); tok.padding_side = "left"; model.config.tokenizer_padding_side = "left"
rng = np.random.default_rng(0)
imgs = [Image.fromarray(rng.integers(0, 255, (336, 336, 3), dtype=np.uint8)) for _ in range(2)]
qs = ["What is in the image?", "Describe the colours and any text that you can see in this picture in detail."]


def prompt(q):
    c = conv_templates["vicuna_v1"].copy(); c.append_message(c.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + q); c.append_message(c.roles[1], None)
    return tokenizer_image_token(c.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt")


def run(idx):
    ids = [prompt(qs[i]) for i in idx]
    L = max(len(x) for x in ids)
    inp = torch.stack([torch.cat([torch.full((L - len(x),), tok.pad_token_id or 0), x]) for x in ids]).cuda()
    am = torch.stack([torch.cat([torch.zeros(L - len(x)), torch.ones(len(x))]) for x in ids]).long().cuda()
    im = process_images([imgs[i] for i in idx], proc, model.config).to("cuda", torch.float16)
    out = model.generate(inp, attention_mask=am, images=im, do_sample=False, max_new_tokens=24, use_cache=True)
    return [tok.decode(o, skip_special_tokens=True) for o in out]


ref = [run([i])[0] for i in range(2)]
enable_fastv(model, 2, 0.0)
r0 = [run([i])[0] for i in range(2)]
print("1. ratio=0 identical to unpatched:", r0 == ref, r0)
st = model.model._fastv
st["ratio"] = 0.75
r75 = [run([i])[0] for i in range(2)]
keep = st["keep"]
print("2. ratio=0.75 kept visual tokens per sample:", [(int(keep[b].sum()) - (keep.shape[1] - 576)) for b in range(keep.shape[0])], "output changed:", r75 != ref, r75)
rb = run([0, 1])
print("3. batched == single at ratio 0.75:", rb == r75, rb)
print("SELFTEST", "PASS" if (r0 == ref and rb == r75) else "FAIL")

"""FastV (Chen et al., ECCV 2024) as a training-free visual-token pruning baseline, accuracy-faithful.

FastV lets the first K LM layers see every visual token, scores each visual token by the attention
the last prompt token pays to it in layer K-1 (mean over heads), and drops the lowest-scoring
fraction `ratio` of them before layer K.  Here the drop is implemented as a key mask: layers >= K
cannot attend to the dropped tokens (neither at prefill nor while decoding).  A masked key
contributes exactly zero to every softmax, so the logits equal those of physically removing the
tokens (positions are unchanged in both, as in the FastV reference code), and it works with the
HF KV cache.  It does NOT reproduce FastV's latency saving: keys/values of dropped tokens are
still computed.  Latency is estimated analytically or measured with a physically pruned build.

Inference only; needs attn_implementation="eager" so that layer K-1's attention row is available
and the 4D causal mask is always materialised.

    enable_fastv(model, k=2, ratio=0.75)   # model: LlavaLlamaForCausalLM
"""
import types

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from llava.constants import IMAGE_TOKEN_INDEX


def enable_fastv(model, k: int, ratio: float):
    if model.config._attn_implementation != "eager":
        raise ValueError("FastV needs attn_implementation='eager' (got "
                         f"{model.config._attn_implementation!r}).")
    assert 0 <= ratio < 1 and 0 < k < len(model.model.layers)
    state = {"k": k, "ratio": ratio, "vis_mask": None, "keep": None}
    model.model._fastv = state
    model.model.forward = types.MethodType(_fastv_forward, model.model)

    orig_prepare = model.prepare_inputs_labels_for_multimodal

    def prepare(input_ids, position_ids, attention_mask, past_key_values, labels, images, *a, **kw):
        out = orig_prepare(input_ids, position_ids, attention_mask, past_key_values, labels, images, *a, **kw)
        embeds, am_out = out[4], out[2]
        if embeds is None or images is None:
            return out
        bsz, seq = embeds.shape[:2]
        am_in = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        left = getattr(model.config, "tokenizer_padding_side", "right") == "left"
        vis = torch.zeros(bsz, seq, dtype=torch.bool, device=embeds.device)
        for b in range(bsz):
            ids = input_ids[b][am_in[b]]
            pos = (ids == IMAGE_TOKEN_INDEX).nonzero().flatten()
            assert pos.numel() == 1, "FastV baseline supports exactly one image per prompt"
            cur_len = seq if am_out is None else int(am_out[b].sum())
            n_vis = cur_len - (ids.numel() - 1)
            start = (seq - cur_len if left else 0) + int(pos[0])
            vis[b, start:start + n_vis] = True
        state["vis_mask"], state["keep"] = vis, None
        return out

    model.prepare_inputs_labels_for_multimodal = prepare
    return model


def _fastv_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                   inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                   return_dict=None, cache_position=None):
    """transformers 4.44.2 LlamaModel.forward with the FastV key mask added (inference only)."""
    st = self._fastv
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, DynamicCache):
        return_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    if cache_position is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    prefill = inputs_embeds.shape[1] > 1 and st["vis_mask"] is not None and st["vis_mask"].shape[1] == inputs_embeds.shape[1]
    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, False)
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    min_val = torch.finfo(hidden_states.dtype).min
    next_cache = None

    for i, layer in enumerate(self.layers):
        if i == st["k"] and st["keep"] is not None:
            keep = st["keep"]
            if keep.shape[1] < causal_mask.shape[-1]:     # decode steps: tokens generated since prefill stay visible
                keep = torch.cat([keep, keep.new_ones(keep.shape[0], causal_mask.shape[-1] - keep.shape[1])], 1)
            layer_mask = causal_mask.masked_fill(~keep[:, None, None, :], min_val)
        elif i < st["k"] or st["keep"] is None:
            layer_mask = causal_mask
        want_attn = prefill and i == st["k"] - 1
        out = layer(hidden_states, attention_mask=layer_mask, position_ids=position_ids,
                    past_key_value=past_key_values, output_attentions=want_attn, use_cache=use_cache,
                    cache_position=cache_position, position_embeddings=position_embeddings)
        hidden_states = out[0]
        if use_cache:
            next_cache = out[-1]
        if want_attn:
            st["keep"] = _fastv_keep(out[1], attention_mask, st)
            st["vis_mask"] = None
            layer_mask = None  # recomputed at layer k from st["keep"]

    hidden_states = self.norm(hidden_states)
    if return_legacy_cache and next_cache is not None:
        next_cache = next_cache.to_legacy_cache()
    if not return_dict:
        return tuple(v for v in (hidden_states, next_cache) if v is not None)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache)


def _fastv_keep(attn, attention_mask, st):
    """attn: layer K-1 weights [B, H, L, L].  Returns keep [B, L]: all non-visual tokens plus the top-scoring visual ones."""
    bsz, _, seq, _ = attn.shape
    vis = st["vis_mask"]
    if attention_mask is None:
        last = torch.full((bsz,), seq - 1, device=attn.device)
    else:
        last = (attention_mask[:, :seq].long() * torch.arange(seq, device=attn.device)).argmax(1)
    score = attn[torch.arange(bsz, device=attn.device), :, last, :].float().mean(1)   # [B, L]
    keep = ~vis
    for b in range(bsz):
        idx = vis[b].nonzero().flatten()
        n_keep = int(round(idx.numel() * (1.0 - st["ratio"])))
        if n_keep > 0:
            keep[b, idx[score[b, idx].topk(n_keep).indices]] = True
    return keep

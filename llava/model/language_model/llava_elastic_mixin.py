"""Shared elastic forward logic for all FlexLLaVA language model backbones.

LlavaElasticMixin provides forward_single_matryoshka, forward, generate, and
prepare_inputs_for_generation so that each backbone file (llava_qwen.py,
llava_phi.py, etc.) stays minimal.  Only the backbone-specific __init__ and
the AutoConfig/AutoModelForCausalLM registrations differ per file.

MRO contract: this mixin must be listed BEFORE the HF backbone class so that
super().forward() inside this mixin resolves to the correct backbone forward.
Example:
    class LlavaQwenForCausalLM(LlavaElasticMixin, Qwen2ForCausalLM, LlavaMetaForCausalLM):
        ...
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from torch.nn import CrossEntropyLoss


class LlavaElasticMixin:
    """Drop-in mixin that gives any LlavaXxxForCausalLM the elastic grid forward."""

    # ------------------------------------------------------------------
    # Single-level forward (one tok-budget, one LoRA level)
    # ------------------------------------------------------------------
    def forward_single_matryoshka(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        return_dict=None,
        matryoshka_vis_token_scale=None,
    ):
        if inputs_embeds is None:
            (
                input_ids, position_ids, attention_mask,
                past_key_values, inputs_embeds, labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values,
                labels, images, image_sizes,
                matryoshka_vis_token_scale=matryoshka_vis_token_scale,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        # Works for all backbones; Llama's pretraining_tp > 1 path is rarely
        # used in practice and not needed for small LLMs.
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return loss, logits, outputs

    # ------------------------------------------------------------------
    # Full forward: elastic grid / pure-M3 / standard eval
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        return_dict=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # ---- Elastic (Adaptive Matryoshka) training branch --------------------
        engine = getattr(self, "elastic_engine", None)
        if self.training and engine is not None:
            from llava.model.elastic import losses as _el
            loss = 0
            ce_total = 0.0
            kl_total = 0.0
            coral_total = 0.0
            per_level = {}
            logits_accumulate = []
            grid = list(engine.grid())
            teacher_logits = None
            teacher_tokens = None
            for l_tok in grid:
                engine._set_lora_level(l_tok)
                loss_item, logits, outputs = self.forward_single_matryoshka(
                    input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, labels=labels,
                    use_cache=use_cache, output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    images=images, image_sizes=image_sizes, return_dict=return_dict,
                    matryoshka_vis_token_scale=l_tok,
                )
                n_tok = engine.cfg.tok_levels[l_tok]
                tag = f"tok{n_tok}"
                ce_val = loss_item.item() / len(grid)
                ce_total += ce_val
                per_level[f"loss/ce_{tag}"] = loss_item.item()
                loss += loss_item / len(grid)
                cur_tokens = engine.last_tokens
                if l_tok == engine.cfg.kl_teacher_tok_level:
                    teacher_logits = logits.detach()
                    teacher_tokens = cur_tokens.detach() if cur_tokens is not None else None
                else:
                    if engine.cfg.use_prefix_kl and teacher_logits is not None:
                        n = min(teacher_logits.shape[1], logits.shape[1])
                        s_log = logits[:, logits.shape[1] - n:]
                        t_log = teacher_logits[:, teacher_logits.shape[1] - n:]
                        kl_labels = None
                        if labels is not None:
                            L = min(n, labels.shape[1])
                            kl_labels = labels[:, labels.shape[1] - L:]
                        kl = _el.prefix_kl_loss(s_log, t_log, kl_labels)
                        kl_weighted = engine.cfg.prefix_kl_weight * kl / len(grid)
                        kl_total += kl_weighted.item()
                        per_level[f"loss/kl_{tag}"] = kl.item()
                        loss = loss + kl_weighted
                    if (engine.cfg.use_coral_align and teacher_tokens is not None
                            and cur_tokens is not None):
                        coral = _el.coral_loss(cur_tokens, teacher_tokens)
                        coral_weighted = engine.cfg.coral_weight * coral / len(grid)
                        coral_total += coral_weighted.item()
                        per_level[f"loss/coral_{tag}"] = coral.item()
                        loss = loss + coral_weighted
                logits_accumulate.append(logits)
            self._loss_components = {
                "loss/ce": ce_total, "loss/kl": kl_total, "loss/coral": coral_total,
                **per_level,
            }
            engine.maybe_log_adapters()
            logits = torch.cat(logits_accumulate, dim=1)
            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output
            return CausalLMOutputWithPast(
                loss=loss, logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            )

        # ---- Pure M3 (no elastic engine, explicit scale list) -----------------
        if self.training and self.config.matryoshka_vis_token_scale is not None:
            loss = 0
            logits_accumulate = []
            for scale in self.config.matryoshka_vis_token_scale:
                loss_item, logits, outputs = self.forward_single_matryoshka(
                    input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, labels=labels,
                    use_cache=use_cache, output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    images=images, image_sizes=image_sizes, return_dict=return_dict,
                    matryoshka_vis_token_scale=scale,
                )
                loss += loss_item / len(self.config.matryoshka_vis_token_scale)
                logits_accumulate.append(logits)
            logits = torch.cat(logits_accumulate, dim=1)
            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output
            return CausalLMOutputWithPast(
                loss=loss, logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            )

        # ---- Standard eval / non-matryoshka forward ---------------------------
        if inputs_embeds is None:
            (
                input_ids, position_ids, attention_mask,
                past_key_values, inputs_embeds, labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask,
                past_key_values, labels, images, image_sizes,
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        inputs=None,
        images=None,
        image_sizes=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        matryoshka_vis_token_scale = kwargs.pop("matryoshka_vis_token_scale", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _
             ) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None,
                images, image_sizes=image_sizes,
                matryoshka_vis_token_scale=matryoshka_vis_token_scale,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs

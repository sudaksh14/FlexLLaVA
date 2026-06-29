#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def forward_single_matryoshka(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        matryoshka_vis_token_scale: Optional[int] = None,
    ):
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                matryoshka_vis_token_scale = matryoshka_vis_token_scale,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
        return loss, logits, outputs



    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # ---- Elastic (Adaptive Matryoshka) training branch -------------
        # Single axis (L_tok). LoRA level (if any) is set per tok level BEFORE
        # forward_single_matryoshka, whose internal re-encode then uses the
        # right adapter. Pure M3 (no engine) runs the original path below.
        engine = getattr(self, "elastic_engine", None)
        if self.training and engine is not None:
            from ..elastic import losses as _el
            loss = 0
            ce_total = 0.0
            kl_total = 0.0
            coral_total = 0.0
            per_level = {}   # keyed by actual token count, e.g. "tok256"
            logits_accumulate = []
            grid = list(engine.grid())
            teacher_logits = None
            teacher_tokens = None
            for l_tok in grid:
                engine._set_lora_level(l_tok)
                loss_item, logits, outputs = self.forward_single_matryoshka(
                    input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    images=images, image_sizes=image_sizes, return_dict=return_dict,
                    matryoshka_vis_token_scale=l_tok,
                )
                n_tok = engine.cfg.tok_levels[l_tok]
                tag = f"tok{n_tok}"
                ce_val = loss_item.item() / len(grid)
                ce_total += ce_val
                per_level[f"loss/ce_{tag}"] = loss_item.item()   # raw CE at this level
                loss += loss_item / len(grid)
                cur_tokens = engine.last_tokens
                if l_tok == engine.cfg.kl_teacher_tok_level:
                    teacher_logits = logits.detach()
                    teacher_tokens = cur_tokens.detach() if cur_tokens is not None else None
                else:
                    if engine.cfg.use_prefix_kl and teacher_logits is not None:
                        n = min(teacher_logits.shape[1], logits.shape[1])
                        # Tail-align: visual tokens are at the start, text at the end.
                        # Taking the last n positions of each gives comparable text
                        # positions even though teacher has more visual tokens.
                        s_log = logits[:, logits.shape[1] - n:]
                        t_log = teacher_logits[:, teacher_logits.shape[1] - n:]
                        # labels is the original (pre-expansion) text sequence;
                        # clip to n for safety in case it's shorter.
                        kl_labels = None
                        if labels is not None:
                            L = min(n, labels.shape[1])
                            kl_labels = labels[:, labels.shape[1] - L:]
                        kl = _el.prefix_kl_loss(s_log, t_log, kl_labels)
                        kl_weighted = engine.cfg.prefix_kl_weight * kl / len(grid)
                        kl_total += kl_weighted.item()
                        per_level[f"loss/kl_{tag}"] = kl.item()  # raw KL at this level
                        loss = loss + kl_weighted
                    if (engine.cfg.use_coral_align and teacher_tokens is not None
                            and cur_tokens is not None):
                        coral = _el.coral_loss(cur_tokens, teacher_tokens)
                        coral_weighted = engine.cfg.coral_weight * coral / len(grid)
                        coral_total += coral_weighted.item()
                        per_level[f"loss/coral_{tag}"] = coral.item()  # raw CORAL at this level
                        loss = loss + coral_weighted
                logits_accumulate.append(logits)
            # side-channel for LLaVATrainer.compute_loss to pick up and log
            self._loss_components = {
                "loss/ce": ce_total,
                "loss/kl": kl_total,
                "loss/coral": coral_total,
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

        if self.training and self.config.matryoshka_vis_token_scale is not None:
            # print("The model is in training mode.")
            loss = 0
            logits_accumulate = []
            for matryoshka_vis_token_scale_element in self.config.matryoshka_vis_token_scale:
                loss_item,  logits, outputs = self.forward_single_matryoshka(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    position_ids = position_ids,
                    past_key_values = past_key_values,
                    inputs_embeds = inputs_embeds,
                    labels = labels,
                    use_cache = use_cache,
                    output_attentions = output_attentions,
                    output_hidden_states = output_hidden_states,
                    images = images,
                    image_sizes = image_sizes,
                    return_dict = return_dict,
                    matryoshka_vis_token_scale= matryoshka_vis_token_scale_element
                )
                loss += loss_item/len(self.config.matryoshka_vis_token_scale)
                logits_accumulate.append(logits)
                assert len(outputs) == 1, 'len(outputs) == 1 is False'
            logits = torch.cat(logits_accumulate, dim = 1)
                    
            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            
        else:
            # print("The model is in evaluation mode or trained without matryoshka_vis_token_scale.")
            
        
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
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
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        matryoshka_vis_token_scale = kwargs.pop("matryoshka_vis_token_scale", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                matryoshka_vis_token_scale = matryoshka_vis_token_scale
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

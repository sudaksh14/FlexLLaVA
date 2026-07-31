import os
import torch
import torch.nn as nn

from torch.utils.data import Dataset, Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf
from typing import List, Optional


def _patch_bf16_optimizer_overflow_guard():
    """DeepSpeed's fp16 ZeRO optimizer skips the update and clears grads when it
    detects a non-finite gradient (has_overflow/_update_scale). Its BF16_Optimizer
    has no equivalent: bf16_optimizer.py's step() computes the global grad norm,
    asserts it is > 0 (which raises if the norm is NaN, but silently proceeds --
    corrupting every parameter via clip_tensors_by_global_norm -- if it's +inf,
    since 1/inf clips to 0... except when a handful of individual grads are NaN
    without pushing the norm itself into a comparison that trips the assert).
    Bottom line: this path was seen letting a single bad micro-batch permanently
    NaN the whole model (see run_26204, step 54) with no crash and no skip.
    Patch step() to explicitly check the norm for finiteness first and, if not
    finite, clear gradients and skip the update -- mirroring the fp16 behavior.
    """
    from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
    from deepspeed.runtime.utils import get_global_norm_of_tensors, clip_tensors_by_global_norm

    @torch.no_grad()
    def guarded_step(self, closure=None):
        if closure is not None:
            raise NotImplementedError(f'{self.__class__} does not support closure.')

        all_groups_norm = get_global_norm_of_tensors(input_tensors=self.get_grads_for_norm(),
                                                       mpu=self.mpu,
                                                       norm_type=self.norm_type,
                                                       use_graph=self.graph_harvesting)
        if not torch.isfinite(torch.as_tensor(float(all_groups_norm))):
            print(f"[bf16-guard] non-finite global grad norm ({all_groups_norm}) -- "
                  f"skipping optimizer step and clearing gradients instead of "
                  f"applying/crashing.", flush=True)
            self.clear_hp_grads()
            return

        self._global_grad_norm = all_groups_norm
        assert all_groups_norm > 0.
        if self.clip_grad > 0.:
            clip_tensors_by_global_norm(input_tensors=self.get_grads_for_norm(for_clipping=True),
                                         max_norm=self.clip_grad,
                                         global_norm=all_groups_norm,
                                         mpu=self.mpu,
                                         use_graph=self.graph_harvesting)

        self.optimizer.step()
        self.update_lp_params()
        self.clear_hp_grads()

    BF16_Optimizer.step = guarded_step


_patch_bf16_optimizer_overflow_guard()


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult = 8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps, # TODO: seems that this may work?
                lengths=lengths,
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps, # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_modality_length_auto:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps, # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality_auto=True,
            )
        elif self.args.group_by_varlen:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                # self.args.train_batch_size, # TODO: seems that we should have gradient_accumulation_steps
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps, # TODO: seems that this may work?
                lengths=lengths,
                variable_length=True
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper['mm_projector'] = self.args.mm_projector_lr
                # The elastic connector is named elastic_projector/elastic_resampler
                # (the base mm_projector is dropped), so the 'mm_projector' keyword
                # never matches it. Route the elastic connector to the same LR the
                # recipe intends for the projector, otherwise it falls back to the
                # main (LLM-LoRA) learning rate instead of the lower connector LR.
                lr_mapper['elastic_projector'] = self.args.mm_projector_lr
                lr_mapper['elastic_resampler'] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper['vision_tower'] = self.args.mm_vision_tower_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": lr,
                        },
                    ])
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    @staticmethod
    def _elastic_owner(model):
        """Find the module the elastic forward actually ran as.

        attach_elastic_engine() deliberately attaches `elastic_engine` (and the
        forward then sets `_loss_components`) on the *underlying*
        LlavaXxxForCausalLM, not on any wrapper -- see its docstring. By the
        time compute_loss() sees the model it may be wrapped as
        DeepSpeedEngine(PeftModel(LoraModel(LlavaLlamaForCausalLM))).

        Why this exists: elastic loss components were logged in the pre-LoRA
        runs of 2026-07-17 and silently vanished from 2026-07-22 onwards, the
        moment --lora_enable True was introduced (verified in the wandb
        binaries: `grad_norm` logged 2092 times in run_26233, `loss/ce` zero
        times). That cost us the per-term breakdown for the whole TinyLlama
        divergence, which is exactly when it was needed.

        The correlation is solid; the mechanism is NOT. A plain getattr()
        demonstrably *does* forward through a real peft.get_peft_model wrapper
        (see jobs/verify_component_logging.sh), so the obvious explanation is
        wrong and the true cause remains unidentified. This is therefore a
        belt-and-braces fallback, only consulted when getattr returns None:
        walk the wrapper chain explicitly and match on __dict__, so no
        __getattr__ forwarding is involved either way. Verified against bare /
        DeepSpeed / PEFT / DeepSpeed+PEFT / real-PEFT, plus a no-engine
        negative case and a self-referential wrapper.
        """
        seen, stack = set(), [model]
        while stack:
            obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if "elastic_engine" in vars(obj):
                return obj
            for attr in ("module", "base_model", "model"):
                nxt = getattr(obj, attr, None)
                if isinstance(nxt, torch.nn.Module):
                    stack.append(nxt)
        return None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        # A NaN/Inf loss (e.g. from a log(0) edge case in prefix_kl_loss/
        # coral_loss on a rare batch) must never reach backward() as-is: DeepSpeed's
        # own overflow detection is not reliable with every optimizer (confirmed
        # untested with adamw_bnb_8bit, see its own startup warning), and once a
        # NaN gradient gets applied to a shared weight, every subsequent forward
        # pass on any input produces NaN forever -- permanent, silent corruption
        # for the rest of the run. Replace the loss with a zero-connected dummy
        # (still linked to every trainable param, so backward() populates real
        # zero gradients rather than reusing the poisoned graph) so this batch
        # contributes nothing to the accumulated gradient instead of destroying it.
        if not torch.isfinite(loss):
            import torch.distributed as dist
            if not (dist.is_initialized() and dist.get_rank() != 0):
                print(f"[health] NON-FINITE total loss at step {self.state.global_step} "
                      f"-- skipping this batch's gradient contribution (zeroed, not applied).",
                      flush=True)
            zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
            dummy = sum((p.float().sum() for p in model.parameters() if p.requires_grad), zero)
            loss = dummy * 0.0

        # Log elastic component losses + perplexity if the forward populated them.
        components = getattr(model, "_loss_components", None)
        if components is None:
            # getattr failed to reach through the DeepSpeed/PEFT wrappers --
            # resolve the real owner once and cache it.
            if not hasattr(self, "_elastic_owner_cache"):
                self._elastic_owner_cache = self._elastic_owner(model)
            owner = self._elastic_owner_cache
            if owner is not None:
                components = vars(owner).get("_loss_components")
        # compute_loss() runs once per *micro*-batch, so logging here
        # unthrottled emits gradient_accumulation_steps (32) entries per
        # optimizer step -- 32x the intended rate, which is how run_25726.out
        # reached 117MB. Emit at most one entry per global_step so the elastic
        # components land at the same cadence as HF's own loss logging.
        if components is not None:
            step = self.state.global_step
            if getattr(self, "_last_component_log_step", -1) == step:
                components = None
            else:
                self._last_component_log_step = step
        if components is not None:
            log_dict = dict(components)
            log_dict["loss/total"] = loss.item()
            ce = components.get("loss/ce")
            if ce is not None:
                # Perplexity is only meaningful for the CE term. Clamp before
                # exp(): a diverging run makes exp(ce) overflow to inf, which
                # then poisons the wandb panel for the whole run.
                log_dict["perplexity"] = float(torch.exp(torch.tensor(min(ce, 20.0))))
            self.log(log_dict)

        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

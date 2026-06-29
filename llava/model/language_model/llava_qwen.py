"""FlexLLaVA wrapper for Qwen2 / Qwen2.5 LLM backbones.

Covers: Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct,
        Qwen/Qwen2.5-3B-Instruct (and their base variants).

MRO: LlavaElasticMixin comes first so super().forward() inside the mixin
resolves to Qwen2ForCausalLM.forward(), then to LlavaMetaForCausalLM.
"""

import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

try:
    from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
except ImportError:
    raise ImportError(
        "Qwen2 support requires transformers >= 4.37.0. "
        "Please run: pip install 'transformers>=4.37.0'"
    )

from .llava_elastic_mixin import LlavaElasticMixin
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super().__init__(config)


class LlavaQwenForCausalLM(LlavaElasticMixin, Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # Skip Qwen2ForCausalLM.__init__ to avoid double-initialising self.model;
        # call PreTrainedModel.__init__ directly (same pattern as llava_llama.py).
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)

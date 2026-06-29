"""FlexLLaVA wrapper for StableLM-2 LLM backbone.

Covers: stabilityai/stablelm-2-zephyr-1_6b (StableLmForCausalLM, model_type="stablelm").

Note: TinyLlama-1.1B and SmolLM2-1.7B use LlamaForCausalLM (model_type="llama")
and are handled by the existing llava_llama.py — no new file needed for them.
"""

import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

try:
    from transformers import StableLmConfig, StableLmModel, StableLmForCausalLM
except ImportError:
    raise ImportError(
        "StableLM support requires transformers >= 4.40.0. "
        "Please run: pip install 'transformers>=4.40.0'"
    )

from .llava_elastic_mixin import LlavaElasticMixin
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaStableLMConfig(StableLmConfig):
    model_type = "llava_stablelm"


class LlavaStableLMModel(LlavaMetaModel, StableLmModel):
    config_class = LlavaStableLMConfig

    def __init__(self, config: StableLmConfig):
        super().__init__(config)


class LlavaStableLMForCausalLM(LlavaElasticMixin, StableLmForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaStableLMConfig

    def __init__(self, config):
        super(StableLmForCausalLM, self).__init__(config)
        self.model = LlavaStableLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model


AutoConfig.register("llava_stablelm", LlavaStableLMConfig)
AutoModelForCausalLM.register(LlavaStableLMConfig, LlavaStableLMForCausalLM)

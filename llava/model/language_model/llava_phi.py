"""FlexLLaVA wrapper for Microsoft Phi-2 LLM backbone.

Covers: microsoft/phi-2 (PhiForCausalLM, model_type="phi").
For Phi-3 (model_type="phi3", Phi3ForCausalLM) add a sibling class below
using the same mixin pattern.
"""

import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import PhiConfig, PhiModel, PhiForCausalLM

from .llava_elastic_mixin import LlavaElasticMixin
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaPhiConfig(PhiConfig):
    model_type = "llava_phi"


class LlavaPhiModel(LlavaMetaModel, PhiModel):
    config_class = LlavaPhiConfig

    def __init__(self, config: PhiConfig):
        super().__init__(config)


class LlavaPhiForCausalLM(LlavaElasticMixin, PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaPhiConfig

    def __init__(self, config):
        super(PhiForCausalLM, self).__init__(config)
        self.model = LlavaPhiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model


AutoConfig.register("llava_phi", LlavaPhiConfig)
AutoModelForCausalLM.register(LlavaPhiConfig, LlavaPhiForCausalLM)

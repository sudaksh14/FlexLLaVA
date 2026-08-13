"""FlexLLaVA wrappers for the Microsoft Phi LLM backbones.

Covers:
  * microsoft/phi-2              -- PhiForCausalLM,  model_type="phi"
  * microsoft/Phi-3.5-mini-*     -- Phi3ForCausalLM, model_type="phi3"

Phi-3.5 uses a different chat format from Phi-2 (<|user|> ... <|end|>
<|assistant|> rather than Instruct:/Output:), so it has its own conversation
template registered as "phi3" -- see llava/conversation.py. Pointing it at the
Phi-2 template would not error, it would just silently train and evaluate on
mismatched prompts.

Padding note: Phi-3.5 ships pad_token == eos_token ('<|endoftext|>', id 32000),
which without ensure_distinct_pad_token would delete every EOS from the labels.
It does have a real <unk> (id 0), so the guard's clean fallback applies -- no
embedding resize, unlike Phi-2 whose unk IS its eos.
"""

import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers import PhiConfig, PhiModel, PhiForCausalLM
from transformers import Phi3Config, Phi3Model, Phi3ForCausalLM

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


class LlavaPhi3Config(Phi3Config):
    model_type = "llava_phi3"


class LlavaPhi3Model(LlavaMetaModel, Phi3Model):
    config_class = LlavaPhi3Config

    def __init__(self, config: Phi3Config):
        super().__init__(config)


class LlavaPhi3ForCausalLM(LlavaElasticMixin, Phi3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaPhi3Config

    def __init__(self, config):
        super(Phi3ForCausalLM, self).__init__(config)
        self.model = LlavaPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model


AutoConfig.register("llava_phi", LlavaPhiConfig)
AutoModelForCausalLM.register(LlavaPhiConfig, LlavaPhiForCausalLM)
AutoConfig.register("llava_phi3", LlavaPhi3Config)
AutoModelForCausalLM.register(LlavaPhi3Config, LlavaPhi3ForCausalLM)

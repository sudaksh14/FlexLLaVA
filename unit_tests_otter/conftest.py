"""pytest options for the Otter-pipeline prerun gate.

Mirrors Otter's conftest.py (Otter/conftest.py), which exposes --yaml-path so
`verify_yaml` can shell out to `pytest -m prerun --yaml-path=...`. Ours takes
the backbone too, because the checks that actually matter here (EOS
supervision, pad!=eos, label masking) are tokenizer-dependent.
"""


def pytest_addoption(parser):
    parser.addoption("--mixture-config", action="store",
                     default="configs/otter/mix665k_baseline.yaml",
                     help="Path to the mixture YAML under test")
    parser.addoption("--model-name-or-path", action="store",
                     default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                     help="Backbone whose tokenizer/template to verify against")
    parser.addoption("--conv-version", action="store", default="v1",
                     help="Conversation template key (v1 / chatml / phi / phi3)")
    parser.addoption("--model-max-length", action="store", type=int, default=2048)
    parser.addoption("--pretrain-elastic-path", action="store", default=None)
    parser.addoption("--n-samples", action="store", type=int, default=128)
    parser.addoption("--hf-cache-dir", action="store", default=None)


def pytest_configure(config):
    config.addinivalue_line("markers", "prerun: mark a test as a prerun check.")

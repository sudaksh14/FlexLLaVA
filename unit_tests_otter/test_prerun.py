"""Prerun gate as pytest tests, mirroring Otter's `pytest -m prerun` contract.

Otter aborts a training job when this fails (verify_yaml, Otter/pipeline/train/
train_utils.py:155). scripts/v1_5/finetune_otter_slm.sh does the same via
`python -m llava.data_otter.verify`; this file is the pytest-shaped entry to
the identical checks, for running them by hand or from CI:

    pytest -m prerun unit_tests_otter \
        --mixture-config configs/otter/mix665k_ocr_heavy.yaml \
        --model-name-or-path HuggingFaceTB/SmolLM2-1.7B-Instruct \
        --conv-version chatml
"""

import pytest

from llava.data_otter.verify import build_arg_parser, run_verification


@pytest.fixture(scope="module")
def report(request):
    opt = request.config.getoption
    argv = [
        "--mixture_config", opt("--mixture-config"),
        "--model_name_or_path", opt("--model-name-or-path"),
        "--version", opt("--conv-version"),
        "--model_max_length", str(opt("--model-max-length")),
        "--n_samples", str(opt("--n-samples")),
    ]
    if opt("--pretrain-elastic-path"):
        argv += ["--pretrain_elastic_path", opt("--pretrain-elastic-path")]
    if opt("--hf-cache-dir"):
        argv += ["--cache_dir", opt("--hf-cache-dir")]
    rep = run_verification(build_arg_parser().parse_args(argv))
    print(rep.render())
    return rep


@pytest.mark.prerun
def test_no_failures(report):
    failures = [(n, m) for level, n, m in report.checks if level == "FAIL"]
    assert not failures, "prerun checks failed:\n" + "\n".join(
        f"  {n}: {m}" for n, m in failures)


@pytest.mark.prerun
def test_checks_actually_ran(report):
    """A gate that silently checks nothing is worse than no gate."""
    names = {n for _, n, _ in report.checks}
    for required in ("mixture-config", "mixture-build", "pad!=eos"):
        assert required in names, f"check {required!r} did not run"

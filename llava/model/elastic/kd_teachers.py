"""KD teacher registry + student/teacher compatibility audit.

Why this module exists
----------------------
`engine.attach_kd_teacher` had exactly one teacher hard-coded
(`liuhaotian/llava-v1.5-7b`) and exactly one compatibility check (vocab_size
equality). That was correct but narrow: it answers "will the KL crash?" and not
"is this a meaningful teacher for this student?". Extending the experiment
matrix to SmolLM2 / MobileLLaMA / Qwen forced the wider question, so the
knowledge lives here as data rather than as branches inside the training loop.

What the current KD actually is (audited 2026-09-13, EXPERIMENT_JOURNAL 18a)
---------------------------------------------------------------------------
LOGITS KD ONLY. `losses.prefix_kl_loss` computes
    KL(teacher || student) = sum_V softmax(t) * (log_softmax(t) - log_softmax(s))
over the vocabulary axis, masked to positions where `labels != -100` (i.e. the
assistant response span), averaged over those positions, times T^2. T is
**1.0 and not configurable** -- the call site never passes it. Weight is
`prefix_kl_weight` (0.1), further divided by the number of active levels.

There is NO hidden-state, feature, or attention KD from an external teacher.
CORAL compares projected *visual* tokens and is explicitly self-sourced even
when an external teacher is attached, so teacher hidden_size never enters any
loss. That is why hidden-dim mismatch is NOT a blocker here (LLaVA-7B is 4096,
TinyLlama 2048, and v12 ran) while vocabulary mismatch is fatal: `F.kl_div`
reduces over the vocab axis elementwise, so the two logit tensors must index
the same tokens in the same order.

Compatibility levels
--------------------
DIRECT                 logits KD works as implemented, no changes.
PROJECTED              needs a learned projection. NOT SUPPORTED by any loss in
                       this repo -- listed for completeness, never returned as
                       runnable.
VOCAB_MAPPING_REQUIRED vocabularies differ; a token-id mapping or logit
                       slice/renormalisation would be needed. NOT IMPLEMENTED.
RESPONSE_LEVEL_ONLY    logit KD inappropriate, but generated-text distillation
                       would work. NOT IMPLEMENTED (no such loss exists here).
INCOMPATIBLE           do not use this teacher for this student.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

DIRECT = "DIRECT"
PROJECTED = "PROJECTED"
VOCAB_MAPPING_REQUIRED = "VOCAB_MAPPING_REQUIRED"
RESPONSE_LEVEL_ONLY = "RESPONSE_LEVEL_ONLY"
INCOMPATIBLE = "INCOMPATIBLE"

# Only DIRECT is runnable today. Everything else requires a loss/adapter this
# repo does not have; returning them as runnable would be the "silent fix"
# the audit is meant to prevent.
RUNNABLE = {DIRECT}


@dataclass
class TeacherSpec:
    """One candidate teacher. Facts here were read off the checkpoint configs
    and tokenizers (jobs/audit_kd_teachers.sh, job 27410), not from model
    names. `loadable_as_llava` records whether it survives the ONE loading
    path attach_kd_teacher has: LlavaLlamaForCausalLM.from_pretrained."""
    key: str
    checkpoint: str
    family: str                  # which student family this teacher belongs to
    vocab_size: int
    hidden_size: int
    architecture: str
    vision_tower: Optional[str]
    mm_projector: Optional[str]
    in_local_cache: bool
    loadable_as_llava: Optional[bool]   # None = not yet smoke-tested
    notes: str = ""


@dataclass
class StudentSpec:
    key: str
    llm_checkpoint: str
    family: str
    vocab_size: int
    hidden_size: int
    conv_template: str


# --- students (vocab/hidden read from HF configs; conv from the launchers) ---
STUDENTS: Dict[str, StudentSpec] = {
    "tinyllama":   StudentSpec("tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                               "llama", 32000, 2048, "v1"),
    "mobilellama": StudentSpec("mobilellama", "mtgv/MobileLLaMA-1.4B-Chat",
                               "mobilellama", 32000, 2048, "v1"),
    "smollm2":     StudentSpec("smollm2", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                               "smollm", 49152, 2048, "chatml"),
    "qwen0.5b":    StudentSpec("qwen0.5b", "Qwen/Qwen2.5-0.5B-Instruct",
                               "qwen", 151936, 896, "chatml"),
    "qwen1.5b":    StudentSpec("qwen1.5b", "Qwen/Qwen2.5-1.5B-Instruct",
                               "qwen", 151936, 1536, "chatml"),
}

TEACHERS: Dict[str, TeacherSpec] = {
    "llava7b": TeacherSpec(
        key="llava7b", checkpoint="liuhaotian/llava-v1.5-7b", family="llama",
        vocab_size=32000, hidden_size=4096, architecture="LlavaLlamaForCausalLM",
        vision_tower="openai/clip-vit-large-patch14-336", mm_projector="mlp2x_gelu",
        in_local_cache=True, loadable_as_llava=True,
        notes="The incumbent. Same CLIP-L/336 vision tower and mlp2x_gelu projector "
              "family as our students, so teacher and student solve the same "
              "multimodal task. Runs at its native 576 visual tokens "
              "(matryoshka_vis_token_scale=None). Used by v12."),
    "mobilevlm2_1.7b": TeacherSpec(
        key="mobilevlm2_1.7b", checkpoint="mtgv/MobileVLM_V2-1.7B", family="mobilellama",
        vocab_size=32000, hidden_size=2048, architecture="MobileLlamaForCausalLM",
        vision_tower="openai/clip-vit-large-patch14-336", mm_projector="ldpnetv2",
        in_local_cache=True, loadable_as_llava=False,
        notes="SMOKE-TESTED 2026-09-13 (job 27411): FAILS to load through "
              "LlavaLlamaForCausalLM.from_pretrained with "
              "'ValueError: Unknown projector type: ldpnetv2' -- this repo's projector "
              "builder has no ldpnetv2. Family-matched VLM for MobileLLaMA; same CLIP-L/336 tower. Its LLM is "
              "MobileLLaMA-1.7B vs the 1.4B student -- only marginally larger, so it "
              "is a weak teacher on capacity grounds even where compatible. "
              "architecture/model_type is mobilevlm with an ldpnetv2 projector, NOT "
              "LlavaLlama+mlp2x_gelu; see loadable_as_llava."),
    "smolvlm": TeacherSpec(
        key="smolvlm", checkpoint="HuggingFaceTB/SmolVLM-Instruct", family="smollm",
        vocab_size=49155, hidden_size=2048, architecture="Idefics3ForConditionalGeneration",
        vision_tower="idefics3 SigLIP (1152-d)", mm_projector="idefics3 connector",
        in_local_cache=True, loadable_as_llava=False,
        notes="SMOKE-TESTED 2026-09-13 (job 27411): FAILS to load through the LLaVA "
              "path -- 'Trying to set a tensor of shape [49155, 2048] in weight (which "
              "has shape [49155, 4096])'. Family VLM for SmolLM2, BUT: vocab 49155 vs the student's 49152 (3 added "
              "image tokens), an Idefics3 architecture the LLaVA loader cannot read, a "
              "SigLIP tower rather than CLIP-L/336, and its tokenizer does not even load "
              "in this environment (tokenizers version: 'data did not match any variant "
              "of untagged enum ModelWrapper'). Three independent blockers."),
}

# Candidate order per student family: family-matched first, then cross-family.
CANDIDATES: Dict[str, List[str]] = {
    "llama":       ["llava7b"],
    "mobilellama": ["mobilevlm2_1.7b", "llava7b"],
    "smollm":      ["smolvlm"],
    "qwen":        [],   # nothing in the local inventory -- see NO_FAMILY_TEACHER
}

# Families with no usable teacher in the local checkpoint inventory. Recorded
# explicitly so auto-selection fails loudly with a reason instead of quietly
# reaching for the 7B and computing a KL over unrelated vocabularies.
NO_FAMILY_TEACHER = {
    "qwen": ("No Qwen-family VLM is present in the local HF cache. Qwen2-VL / "
             "Qwen2.5-VL exist upstream and share the Qwen2 tokenizer, which would "
             "make them the correct candidates -- but nothing has been downloaded or "
             "verified here, and this registry does not list checkpoints it has not "
             "inspected. Note also that Qwen2.5-0.5B's LM head is 151936 wide while "
             "its tokenizer holds 151665 entries; any Qwen KD must confirm the "
             "teacher's head width matches the student's exactly."),
    "smollm": ("SmolVLM is the family VLM but is unusable here for three independent "
               "reasons -- see TEACHERS['smolvlm'].notes."),
}


def check_pair(student_key: str, teacher_key: str) -> dict:
    """Classify one student/teacher pair. Pure function over the registry --
    the empirical inputs (vocab sizes, token-id identity, loadability) were
    established by jobs/audit_kd_teachers.sh and jobs/smoke_kd_teacher_load.sh."""
    s, t = STUDENTS[student_key], TEACHERS[teacher_key]
    reasons: List[str] = []
    level = DIRECT

    if s.vocab_size != t.vocab_size:
        level = VOCAB_MAPPING_REQUIRED
        reasons.append(
            f"vocab {s.vocab_size} vs {t.vocab_size}: prefix_kl_loss reduces over the "
            f"vocab axis elementwise, so the logits must index the same tokens. "
            f"attach_kd_teacher raises on this.")
    if t.loadable_as_llava is False:
        level = INCOMPATIBLE
        reasons.append(
            f"architecture {t.architecture} cannot load through "
            f"LlavaLlamaForCausalLM.from_pretrained, the only path attach_kd_teacher has.")
    elif t.loadable_as_llava is None:
        reasons.append("loadability through the LLaVA path NOT yet smoke-tested.")
    if not t.in_local_cache:
        level = INCOMPATIBLE
        reasons.append("checkpoint not in the local HF cache.")
    if s.family != t.family:
        reasons.append(f"cross-family ({s.family} student, {t.family} teacher) -- "
                       f"acceptable only because the tokenizers are byte-identical.")
    # hidden_size deliberately NOT a gate: no loss in this repo consumes teacher
    # hidden states, so a mismatch is irrelevant to logits KD.
    if s.hidden_size != t.hidden_size:
        reasons.append(f"hidden {s.hidden_size} vs {t.hidden_size} -- irrelevant for "
                       f"logits KD (no hidden-state loss exists here).")
    return {"student": student_key, "teacher": teacher_key, "compatibility": level,
            "runnable_today": level in RUNNABLE, "reasons": reasons}


def resolve_teacher(student_key: str, requested: str = "auto") -> TeacherSpec:
    """Resolve --kd_teacher into a concrete TeacherSpec, or raise.

    Never silently substitutes: if the requested or auto-selected teacher is not
    runnable for this student, this raises with the audit's reason attached.
    """
    if student_key not in STUDENTS:
        raise ValueError(f"unknown student backbone {student_key!r}; "
                         f"known: {sorted(STUDENTS)}")
    s = STUDENTS[student_key]

    if requested not in ("auto", None):
        if requested not in TEACHERS:
            raise ValueError(f"unknown --kd_teacher {requested!r}; "
                             f"known: {sorted(TEACHERS)} or 'auto'")
        res = check_pair(student_key, requested)
        if not res["runnable_today"]:
            raise ValueError(
                f"--kd_teacher {requested!r} is {res['compatibility']} for student "
                f"{student_key!r} and will not be used.\n  " +
                "\n  ".join(res["reasons"]) +
                "\nPass a compatible teacher or --use_kd False. This is refused rather "
                "than substituted so a run can never report KD against a teacher it "
                "did not actually use.")
        return TEACHERS[requested]

    for cand in CANDIDATES.get(s.family, []):
        if check_pair(student_key, cand)["runnable_today"]:
            return TEACHERS[cand]

    why = NO_FAMILY_TEACHER.get(s.family, "no candidate passed the compatibility check")
    tried = CANDIDATES.get(s.family, [])
    detail = "\n".join(
        f"  - {c}: {check_pair(student_key, c)['compatibility']}: "
        f"{'; '.join(check_pair(student_key, c)['reasons'])}" for c in tried)
    raise ValueError(
        f"--kd_teacher auto found no runnable teacher for student {student_key!r} "
        f"(family {s.family!r}).\n{why}\n"
        + (f"Candidates tried:\n{detail}\n" if tried else "")
        + "Run with --use_kd False (self-distillation only), or add a verified "
          "teacher to kd_teachers.TEACHERS. Refusing rather than falling back.")


def audit_all() -> dict:
    """Machine-readable compatibility report over every student x teacher pair."""
    pairs = [check_pair(s, t) for s in STUDENTS for t in TEACHERS]
    auto = {}
    for s in STUDENTS:
        try:
            auto[s] = resolve_teacher(s, "auto").key
        except ValueError as e:
            auto[s] = f"NONE ({str(e).splitlines()[0]})"
    return {
        "kd_mechanism": {
            "type": "logits KD only (prefix-KL)",
            "loss": "KL(teacher||student) over vocab, masked to labels != -100",
            "temperature": 1.0,
            "temperature_configurable": False,
            "weight": "prefix_kl_weight (default 0.1), divided by n_active levels",
            "positions": "assistant-response tokens only, right-aligned",
            "teacher_visual_budget": "native/full (matryoshka_vis_token_scale=None) "
                                     "= 576 tokens for CLIP-L/336 teachers",
            "hidden_state_kd": False,
            "attention_kd": False,
            "feature_kd": "CORAL exists but is SELF-sourced even with an external "
                          "teacher; teacher hidden states are never used",
        },
        "students": {k: asdict(v) for k, v in STUDENTS.items()},
        "teachers": {k: asdict(v) for k, v in TEACHERS.items()},
        "pairs": pairs,
        "auto_resolution": auto,
        "no_family_teacher": NO_FAMILY_TEACHER,
    }

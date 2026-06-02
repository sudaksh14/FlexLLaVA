"""Losses for the nested-query method (pooling-free).

All terms are optional and toggled via ElasticConfig. The LM cross-entropy is
computed in the training loop (as in M3); these are the *additional* matryoshka
terms that shape the nested token ordering and keep the latent space stable.

  prefix_kl    : KL( p(y | full tokens) || p(y | n_k tokens) )  -- coarse->fine
  coral_align  : match mean+covariance of projected tokens across levels
  recon        : reconstruct dropped queries from the kept prefix
  decorrelation: penalize redundancy among retained query tokens

`nested dropout` is not a loss; it is applied in the loop by randomly choosing
the truncation length n_k each step, which (with shared weights) induces the
importance ordering for free.
"""

import torch
import torch.nn.functional as F


def prefix_kl_loss(student_logits, teacher_logits, labels=None, T=1.0):
    """KL between full-token (teacher) and truncated (student) next-token dists.
    logits: (B, L, V). Only positions with valid labels contribute if given."""
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(s, t, reduction="none").sum(-1)  # (B, L)
    if labels is not None:
        mask = (labels != -100).float()
        kl = (kl * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        kl = kl.mean()
    return (T * T) * kl


def coral_loss(feat_student, feat_teacher):
    """CORAL: align second-order statistics of token features across levels.
    feat_*: (B, n, d). Compares pooled mean + covariance, so different token
    counts are comparable. Keeps the LLM-input distribution stable per level."""
    def stats(x):
        x = x.reshape(-1, x.shape[-1])
        mu = x.mean(0)
        xc = x - mu
        cov = (xc.t() @ xc) / max(x.shape[0] - 1, 1)
        return mu, cov
    mu_s, cov_s = stats(feat_student)
    mu_t, cov_t = stats(feat_teacher)
    d = feat_student.shape[-1]
    return ((mu_s - mu_t) ** 2).mean() + ((cov_s - cov_t) ** 2).sum() / (d * d)


def recon_loss(kept_prefix, full_tokens, decoder):
    """Reconstruct full query set from the kept prefix via a small decoder.
    kept_prefix: (B, n_k, d); full_tokens: (B, N, d)."""
    pred = decoder(kept_prefix, full_tokens.shape[1])  # (B, N, d)
    return F.mse_loss(pred, full_tokens)


def decorrelation_loss(tokens):
    """Penalize off-diagonal correlation among retained tokens (encourage each
    token to add information). tokens: (B, n, d)."""
    B, n, d = tokens.shape
    x = tokens - tokens.mean(1, keepdim=True)
    x = F.normalize(x, dim=-1)
    gram = torch.bmm(x, x.transpose(1, 2))  # (B, n, n)
    eye = torch.eye(n, device=tokens.device).unsqueeze(0)
    off = gram * (1 - eye)
    return (off ** 2).sum() / (B * n * max(n - 1, 1))

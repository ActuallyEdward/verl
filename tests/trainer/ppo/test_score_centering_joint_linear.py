# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare joint SC with the original CUDA path through an unfused linear head."""

import json
import os

import pytest
import torch

from verl.trainer.ppo.score_centering import (
    sampled_and_topk_log_probs_from_logits,
    score_centering_auto_correction,
    score_centering_weight_fn,
    topk_log_probs_from_logits,
)
from verl.utils.torch_functional import FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE, logprobs_from_logits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA comparison requires GPU")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [0.7, 1.3])
@pytest.mark.parametrize("rule,threshold", [(None, 2.0), ("token", 2.0), ("token", "0.5_5.0")])
@pytest.mark.parametrize("vocab", [257, 151936])
def test_joint_linear_against_original(dtype, temperature, rule, threshold, vocab):
    # Fail rather than silently comparing against the fallback implementation.
    assert FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE
    assert os.environ.get("VERL_DISABLE_FLASH_ATTN_CE", "0") != "1"
    torch.manual_seed(2026)
    n, hidden_dim, k = 17, 64, 8
    hidden = torch.randn(n, hidden_dim, device="cuda", dtype=dtype)
    weight = (torch.randn(vocab, hidden_dim, device="cuda") / hidden_dim**0.5).to(dtype)
    ids = torch.arange(k, device="cuda").expand(n, -1)
    complete = torch.arange(n, device="cuda") % 2 == 0
    labels = torch.where(complete, torch.arange(n, device="cuda") % (k - 1), k + 3)
    # Explicit zero-probability support slot; sampled labels remain supported.
    q = torch.randn(n, vocab, device="cuda").softmax(-1)
    q[complete, k - 1:] = 0
    q /= q.sum(-1, keepdim=True)
    q_head = q.gather(-1, ids).log()
    q_sample = q.gather(-1, labels[:, None]).squeeze(-1).log()
    advantages = torch.linspace(-1, 1, n, device="cuda")
    mask = torch.ones(n, device="cuda")
    mask[[1, 6, 14]] = 0
    weighting = score_centering_weight_fn(rule, threshold)

    def evaluate(mode):
        h = hidden.detach().clone().requires_grad_()
        w = weight.detach().clone().requires_grad_()
        logits = torch.nn.functional.linear(h, w)
        scaled = logits / temperature
        before = scaled.detach().clone()
        if mode == "joint":
            sampled, head = sampled_and_topk_log_probs_from_logits(scaled, labels, ids, chunk_size=5)
        elif mode == "separate":
            sampled = logprobs_from_logits(scaled, labels, inplace_backward=False)
            head = topk_log_probs_from_logits(scaled, ids, chunk_size=5)
        else:
            dense = scaled.float().log_softmax(-1)
            sampled = dense.gather(-1, labels[:, None]).squeeze(-1)
            head = dense.gather(-1, ids)
        correction, shift, _, _ = score_centering_auto_correction(head, q_head, weighting, complete)
        aligned = sampled - shift
        is_weights = weighting((aligned.detach() - q_sample).exp())
        loss = (mask * advantages * (correction - is_weights * aligned)).sum() / mask.sum()
        grads = torch.autograd.grad(loss, (logits, h, w))
        torch.testing.assert_close(scaled, before, atol=0, rtol=0)
        assert (grads[0][mask == 0] == 0).all()
        assert (grads[1][mask == 0] == 0).all()
        return [x.detach().float() for x in (sampled, head, correction, shift, loss, *grads)]

    # Keep the same FP32 matmul policy for all three paths. No optimizer or stochastic model layers.
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        results = {mode: evaluate(mode) for mode in ("joint", "separate", "dense")}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
    names = ["sampled", "head", "correction", "shift", "loss", "grad_logits", "grad_hidden", "grad_weight"]
    metrics = {"dtype": str(dtype), "temperature": temperature, "rule": str(rule),
               "threshold": threshold, "vocab": vocab, "errors": {}}
    for mode in ("separate", "dense"):
        for name, actual, expected in zip(names, results["joint"], results[mode], strict=True):
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            diff = actual - expected
            relative_l2 = (diff.norm() / expected.norm().clamp_min(1e-12)).item()
            maximum = diff.abs().max().item()
            metrics["errors"][mode + "/" + name] = {"max_abs": maximum, "relative_l2": relative_l2}
            if dtype == torch.bfloat16 and name.startswith("grad_"):
                # Two separate BF16 backward paths round before summing; joint rounds once.
                # Bound both global error and the worst component, including near-zero entries.
                assert relative_l2 < 0.015, (name, mode, relative_l2)
                assert maximum <= 0.015 * expected.abs().max().item() + 1e-7, (name, mode, maximum)
            else:
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-4)
    print("SC_LINEAR_RESULT " + json.dumps(metrics), flush=True)

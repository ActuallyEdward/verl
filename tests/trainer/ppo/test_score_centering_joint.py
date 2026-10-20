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
"""Joint sampled/head scoring against dense forward and gradient oracles."""

import pytest
import torch

from verl.trainer.ppo.score_centering import (
    sampled_and_topk_log_probs_from_logits,
    score_centering_auto_correction,
    score_centering_weight_fn,
)

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("used", ["both", "sample", "head"])
def test_joint_log_probs_dense_oracle(device, dtype, used):
    torch.manual_seed(42)
    logits = torch.randn(19, 7, device=device, dtype=dtype).t().requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    ids = torch.arange(5, device=device).expand(7, -1)
    labels = torch.tensor([0, 1, 7, 3, 10, 15, 4], device=device)
    sampled, head = sampled_and_topk_log_probs_from_logits(logits, labels, ids, chunk_size=3)
    assert sampled.is_contiguous(), "Sampled log-probs must be suitable for NCCL all-gather."
    dense = reference.float().log_softmax(-1)
    expected_sample = dense.gather(-1, labels[:, None]).squeeze(-1)
    expected_head = dense.gather(-1, ids)
    torch.testing.assert_close(sampled, expected_sample, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(head, expected_head, atol=1e-6, rtol=1e-6)
    a, b = torch.randn_like(sampled), torch.randn_like(head)
    loss = (sampled * a).sum() if used != "head" else 0
    oracle = (expected_sample * a).sum() if used != "head" else 0
    if used != "sample":
        loss = loss + (head * b).sum()
        oracle = oracle + (expected_head * b).sum()
    actual_grad = torch.autograd.grad(loss, logits)[0]
    expected_grad = torch.autograd.grad(oracle, reference)[0]
    tolerance = 8e-3 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.3])
@pytest.mark.parametrize("rule,threshold", [(None, 2.0), ("token", 2.0), ("token", "0.5_5.0")])
def test_joint_sc_objective_dense_oracle(device, temperature, rule, threshold):
    torch.manual_seed(17)
    logits = torch.randn(7, 11, device=device, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    ids = torch.arange(4, device=device).expand(7, -1)
    labels = torch.tensor([0, 7, 2, 10, 1, 6, 3], device=device)
    complete = torch.tensor([True, False, True, False, True, False, True], device=device)
    q = torch.randn(7, 11, device=device).log_softmax(-1).exp()
    q[0, 3] = 0
    q[complete, 4:] = 0
    q /= q.sum(-1, keepdim=True)
    q_head, q_sample = q[:, :4].log(), q.gather(-1, labels[:, None]).squeeze(-1).log()
    advantages = torch.tensor([0.7, -0.5, 0, 1, -1.2, 0.3, 0], device=device)
    weight = score_centering_weight_fn(rule, threshold)

    def objective(sampled, head):
        correction, shift, _, _ = score_centering_auto_correction(head, q_head, weight, complete)
        aligned = sampled - shift
        weights = weight((aligned.detach() - q_sample).exp())
        return (advantages * (correction - weights * aligned)).mean()

    sampled, head = sampled_and_topk_log_probs_from_logits(logits / temperature, labels, ids, chunk_size=3)
    dense = (reference / temperature).log_softmax(-1)
    loss = objective(sampled, head)
    oracle = objective(dense.gather(-1, labels[:, None]).squeeze(-1), dense.gather(-1, ids))
    torch.testing.assert_close(loss, oracle, atol=2e-6, rtol=2e-5)
    actual_grad = torch.autograd.grad(loss, logits)[0]
    expected_grad = torch.autograd.grad(oracle, reference)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)
    assert (actual_grad[[2, 6]] == 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_empty_joint_rows(device):
    logits = torch.empty(0, 11, device=device, requires_grad=True)
    sampled, head = sampled_and_topk_log_probs_from_logits(
        logits, torch.empty(0, device=device, dtype=torch.long), torch.empty(0, 4, device=device, dtype=torch.long)
    )
    assert sampled.shape == (0,) and head.shape == (0, 4)
    (sampled.sum() + head.sum()).backward()
    assert logits.grad.shape == logits.shape

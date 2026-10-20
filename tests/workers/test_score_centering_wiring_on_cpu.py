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

"""CPU coverage for score centering's FSDP engine wiring."""

from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


@pytest.mark.parametrize("use_remove_padding", [True, False])
@pytest.mark.parametrize("score_centering", [True, False])
def test_prepare_model_outputs_reuses_score_centering_log_probs(use_remove_padding, score_centering):
    """SC supplies sampled log-probs; other paths still use their existing CE kernel."""
    seq_lengths = torch.tensor([3, 2])
    total_nnz, vocab_size = int(seq_lengths.sum()), 8
    cu_seqlens = torch.cat([torch.tensor([0]), seq_lengths.cumsum(0)])
    input_ids = torch.nested.nested_tensor_from_jagged(torch.randint(0, vocab_size, (total_nnz,)), offsets=cu_seqlens)
    output = type("Output", (), {})()
    output_args = {"input_ids_rmpad_rolled": torch.randint(0, vocab_size, (total_nnz,))}
    if use_remove_padding:
        output.logits = torch.randn(1, total_nnz, vocab_size)
        output_args.update(temperature_rmpad=torch.ones(total_nnz), pad_size=0)
    else:
        output.logits = torch.randn(2, 3, vocab_size)
        output_args["temperature"] = torch.ones(2)

    micro_batch = TensorDict({"input_ids": input_ids}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        score_centering=score_centering,
    )
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False

    expected = torch.arange(total_nnz, dtype=torch.float32)

    def logits_processor(student_logits, data, sampled_token_ids):
        torch.testing.assert_close(sampled_token_ids, output_args["input_ids_rmpad_rolled"])
        return {"sc_correction": torch.zeros(student_logits.shape[:2]), "log_probs": expected.unsqueeze(0)}

    with patch(
        "verl.workers.engine.fsdp.transformer_impl.logprobs_from_logits", return_value=torch.zeros(total_nnz)
    ) as mock_logprobs:
        result = FSDPEngineWithLMHead.prepare_model_outputs(engine, output, output_args, micro_batch, logits_processor)

    if score_centering:
        mock_logprobs.assert_not_called()
        torch.testing.assert_close(result["log_probs"].values(), expected)
    else:
        assert mock_logprobs.call_args.kwargs.get("inplace_backward", True)


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_joint_score_centering_engine_gradient(use_remove_padding):
    from functools import partial
    from types import SimpleNamespace

    from verl.trainer.ppo.score_centering import score_centering_auto_correction, score_centering_logits_processor

    torch.manual_seed(31)
    offsets = torch.tensor([0, 3, 5])

    def nested(values):
        return torch.nested.nested_tensor_from_jagged(values, offsets)

    ids = torch.arange(3).expand(5, -1)
    q = torch.tensor([[0.4, 0.3, 0.2]]).expand(5, -1).log()
    labels = torch.tensor([0, 5, 2, 7, 1])
    micro_batch = TensorDict(
        {"input_ids": nested(labels), "rollout_topk_ids": nested(ids), "rollout_topk_log_probs": nested(q)},
        batch_size=[],
    )
    tu.assign_non_tensor(micro_batch, use_remove_padding=use_remove_padding,
                        pad_mode=DatasetPadMode.NO_PADDING, score_centering=True)
    shape = (1, 5, 8) if use_remove_padding else (2, 3, 8)
    logits = torch.randn(shape, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    args = {"input_ids_rmpad_rolled": labels}
    if use_remove_padding:
        args.update(temperature_rmpad=torch.full((5,), 0.7), pad_size=0)
        packed = reference.squeeze(0) / 0.7
    else:
        args["temperature"] = torch.full((2,), 0.7)
        packed = torch.cat((reference[0], reference[1, :2])) / 0.7
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    config = SimpleNamespace(policy_loss=SimpleNamespace(rollout_correction={}))
    processor = partial(score_centering_logits_processor, config=config)
    result = engine.prepare_model_outputs(SimpleNamespace(logits=logits), args, micro_batch, processor)
    dense = packed.log_softmax(-1)
    sampled = dense.gather(-1, labels[:, None]).squeeze(-1)
    correction, _, _, _ = score_centering_auto_correction(
        dense.gather(-1, ids), q, torch.ones_like, torch.zeros(5, dtype=torch.bool)
    )
    actual = result["sc_correction"].values() - result["log_probs"].values()
    expected = correction - sampled
    torch.testing.assert_close(actual, expected)
    weights = torch.tensor([1., -0.5, 0., 0.3, 0.])
    actual_grad = torch.autograd.grad((actual * weights).sum(), logits)[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), reference)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)

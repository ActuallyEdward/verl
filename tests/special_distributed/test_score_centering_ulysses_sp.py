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
"""Two-GPU SC output/gradient checks, without a model or training loop.

Run: torchrun --standalone --nproc_per_node=2 -m pytest -q -s <this file>
The real FSDP output processor and Ulysses collectives are exercised. Parameter
synchronization is an explicit average, not a claim of full FSDP training coverage.
"""

import os
from datetime import timedelta
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.trainer.ppo.score_centering import (
    score_centering_auto_correction,
    score_centering_logits_processor,
    score_centering_weight_fn,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import entropy_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, set_ulysses_sequence_parallel_group, slice_input_tensor
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


@pytest.fixture(scope="module", autouse=True)
def distributed():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("Launch with torchrun --nproc_per_node=2")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    yield
    set_ulysses_sequence_parallel_group(None)
    dist.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lengths", [(3, 5, 8), (3, 5, 9)])
@pytest.mark.parametrize("entropy", [False, True])
@pytest.mark.parametrize("rule,threshold", [(None, 2.0), ("token", 2.0), ("token", "0.5_5.0")])
def test_sc_sequence_parallel_matches_dense(dtype, lengths, entropy, rule, threshold):
    torch.manual_seed(713)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    n, v, d, k = sum(lengths), 257, 32, 8
    world = dist.get_world_size()
    pad = (-n) % world
    offsets = torch.tensor([0, lengths[0], lengths[0] + lengths[1], n], device=device)
    tokens = (torch.arange(n, device=device) * 7 + 11) % v
    # Roll before slicing, exactly as required by the engine's packed-label contract.
    labels = tokens.roll(-1)
    ids = (labels[:, None] + torch.arange(k, device=device)) % v
    complete = torch.arange(n, device=device) % 2 == 0
    q = torch.randn(n, k, device=device).softmax(-1)
    q[complete, -1] = 0
    q /= q.sum(-1, keepdim=True)
    q *= torch.where(complete, 1.0, 0.75)[:, None]
    h = torch.randn(n, d, device=device, dtype=dtype)
    w = (torch.randn(v, d, device=device) / d**0.5).to(dtype)
    temperatures = torch.linspace(0.7, 1.3, n, device=device)
    mask = torch.ones(n, device=device)
    mask[[2, n // 2, n - 1]] = 0
    advantage = torch.linspace(-1, 1, n, device=device)
    weight_fn = score_centering_weight_fn(rule, threshold)

    def nested(x):
        return torch.nested.nested_tensor_from_jagged(x, offsets)

    data = TensorDict({"input_ids": nested(tokens), "rollout_topk_ids": nested(ids),
                      "rollout_topk_log_probs": nested(q.log()), "rollout_support_complete": nested(complete)},
                     batch_size=[])
    tu.assign_non_tensor(data, use_remove_padding=True, score_centering=True,
                        pad_mode=DatasetPadMode.NO_PADDING, calculate_entropy=entropy)
    local_h = slice_input_tensor(h, dim=0).detach().requires_grad_()
    local_w = w.clone().requires_grad_()
    local_logits = torch.nn.functional.linear(local_h, local_w)
    local_labels = slice_input_tensor(labels, dim=0)
    # Padding must not introduce a zero-temperature division.
    local_temp = slice_input_tensor(temperatures - 1, dim=0) + 1
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = True
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False, entropy_from_logits_with_chunking=False)
    engine.compute_entropy_from_logits = entropy_from_logits
    config = SimpleNamespace(policy_loss=SimpleNamespace(rollout_correction={
        "rollout_is": rule, "rollout_is_threshold": threshold,
    }))
    actual = engine.prepare_model_outputs(
        SimpleNamespace(logits=local_logits.unsqueeze(0)),
        {"input_ids_rmpad_rolled": local_labels, "temperature_rmpad": local_temp, "pad_size": pad},
        data, partial(score_centering_logits_processor, config=config),
    )
    ref_h = h.clone().requires_grad_()
    ref_w = w.clone().requires_grad_()
    ref_logits = torch.nn.functional.linear(ref_h, ref_w)
    scaled = ref_logits / temperatures.to(dtype)[:, None]
    dense = scaled.float().log_softmax(-1)
    correction, shift, q_mass, p_mass = score_centering_auto_correction(
        dense.gather(-1, ids), q.log(), weight_fn, complete
    )
    expected = {"log_probs": dense.gather(-1, labels[:, None]).squeeze(-1),
                "sc_correction": correction, "sc_log_support_mass": shift,
                "sc_sampler_head_mass": q_mass, "sc_train_head_mass": p_mass,
                "sc_support_aligned": complete.float()}
    if entropy:
        expected["entropy"] = entropy_from_logits(scaled)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key].offsets(), offsets)
        torch.testing.assert_close(actual[key].values().float(), value.float(), atol=3e-6, rtol=1e-4)

    def objective(outputs):
        aligned = outputs["log_probs"] - outputs["sc_log_support_mass"]
        is_weight = weight_fn((aligned.detach() - q[:, 0].log()).exp())
        per_token = advantage * (outputs["sc_correction"] - is_weight * aligned)
        if entropy:
            per_token = per_token - 0.01 * outputs["entropy"]
        return (mask * per_token).sum() / mask.sum()

    loss = objective({key: actual[key].values() for key in expected})
    reference_loss = objective(expected)
    torch.testing.assert_close(loss, reference_loss, atol=3e-6, rtol=1e-4)
    grads = torch.autograd.grad(loss, (local_logits, local_h, local_w))
    reference_grads = torch.autograd.grad(reference_loss, (ref_logits, ref_h, ref_w))
    # Ulysses Gather backward scales by SP size to compensate for parameter-gradient averaging.
    # Reassemble activation gradients without applying a second autograd scaling.
    errors = []
    for grad, ref in zip(grads[:2], reference_grads[:2], strict=True):
        reconstructed = gather_outputs_and_unpad(grad.float() / world, 0, 0, pad, grad_scaler=False)
        if pad and dist.get_rank() == world - 1:
            assert (grad[-pad:] == 0).all()
        assert (reconstructed[mask == 0] == 0).all()
        errors.append(check_gradient(reconstructed, ref.float(), dtype))
    averaged = grads[2].float()
    dist.all_reduce(averaged)
    averaged /= world
    errors.append(check_gradient(averaged, reference_grads[2].float(), dtype))
    if dist.get_rank() == 0:
        print(f"SC_SP_RESULT dtype={dtype} lengths={lengths} entropy={entropy} rule={rule}/{threshold} "
              f"max_relative_l2={max(errors):.9g}", flush=True)


def check_gradient(actual, expected, dtype):
    assert torch.isfinite(actual).all()
    error = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    if dtype == torch.bfloat16:
        assert error.item() < 0.015
        assert (actual - expected).abs().max() <= expected.abs().max() * 0.015 + 1e-7
    else:
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-4)
    return error.item()

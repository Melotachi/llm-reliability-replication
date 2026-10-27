"""Tests for src/generation.py.

Fast tests mock HF internals; slow tests (@pytest.mark.slow) load the real
codegen-350M-mono model per CLAUDE.md FAQ guidance. Run only fast tests
during normal development: `pytest tests/ -m "not slow"`.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src import generation


def test_truncate_at_stop_sequence_cuts_before_next_function():
    text = "    return True\n" "\n\ndef unrelated_next_function(x):\n" "    return x"
    assert generation._truncate_at_stop_sequence(text) == "    return True\n\n"


def test_truncate_at_stop_sequence_ignores_indented_keywords():
    # An `if` nested inside the target function (indented) must NOT trigger
    # truncation — only a top-level (column-0) keyword should.
    text = "    if x > 0:\n        return True\n    return False\n"
    assert generation._truncate_at_stop_sequence(text) == text


def test_truncate_at_stop_sequence_no_match_returns_unchanged():
    text = "    return 1\n"
    assert generation._truncate_at_stop_sequence(text) == text


def _fake_loaded_model(dtype: torch.dtype = torch.float32) -> MagicMock:
    """A stand-in for a loaded HF model that exposes a real parameter dtype.

    load_model() verifies the dtype of the weights it just loaded, so a bare
    MagicMock (whose .parameters() yields another MagicMock) trips the guard.
    The fake carries one real tensor so the verification exercises the same
    code path the pipeline does.
    """
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1, dtype=dtype)])
    return model


def test_load_model_sets_pad_token_and_moves_to_device():
    fake_tokenizer = MagicMock(pad_token=None, eos_token="<eos>")
    expected_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    fake_model = _fake_loaded_model(expected_dtype)
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"

    with patch.object(
        generation.AutoTokenizer, "from_pretrained", return_value=fake_tokenizer
    ) as tok_load, patch.object(
        generation.AutoModelForCausalLM, "from_pretrained", return_value=fake_model
    ) as model_load:
        tokenizer, model = generation.load_model("Salesforce/codegen-350M-mono")

    tok_load.assert_called_once_with("Salesforce/codegen-350M-mono")
    model_load.assert_called_once()
    assert tokenizer.pad_token == "<eos>"
    fake_model.to.assert_called_once_with(expected_device)
    fake_model.eval.assert_called_once()
    assert (tokenizer, model) == (fake_tokenizer, fake_model)


def test_resolve_device_auto_never_selects_mps():
    # Week 7 N-sweep: "auto" must keep reproducing the pre-Week-7 CPU runs on
    # an Apple-silicon machine, where MPS is available but changes the draws.
    with patch.object(torch.cuda, "is_available", return_value=False):
        assert generation._resolve_device("auto") == "cpu"
    with patch.object(torch.cuda, "is_available", return_value=True):
        assert generation._resolve_device("auto") == "cuda"


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_resolve_device_explicit_is_passed_through(device):
    assert generation._resolve_device(device) == device


def test_resolve_dtype_auto_matches_pre_week7_rule():
    assert generation._resolve_dtype("auto", "cuda") is torch.float16
    assert generation._resolve_dtype("auto", "cpu") is torch.float32
    assert generation._resolve_dtype("auto", "mps") is torch.float32
    # Explicit wins over the device-derived default — this is what pins a GPU
    # run to float32 so it stays comparable with the CPU-generated runs.
    assert generation._resolve_dtype("float32", "cuda") is torch.float32


@pytest.mark.parametrize(
    "kwargs", [{"device": "tpu"}, {"dtype": "int8"}], ids=["bad-device", "bad-dtype"]
)
def test_load_model_rejects_unknown_device_or_dtype(kwargs):
    with pytest.raises(ValueError):
        generation.load_model("Salesforce/codegen-350M-mono", **kwargs)


def test_load_model_honors_explicit_device_and_dtype():
    fake_tokenizer = MagicMock(pad_token="<pad>", eos_token="<eos>")
    fake_model = _fake_loaded_model(torch.float32)

    with patch.object(
        generation.AutoTokenizer, "from_pretrained", return_value=fake_tokenizer
    ), patch.object(
        generation.AutoModelForCausalLM, "from_pretrained", return_value=fake_model
    ) as model_load:
        generation.load_model(
            "Salesforce/codegen-350M-mono", device="cuda", dtype="float32"
        )

    # Either spelling is acceptable — `dtype=` is the modern one and
    # `torch_dtype=` the fallback for older transformers — but the value must
    # be the requested dtype, not the checkpoint's own.
    kwargs = model_load.call_args.kwargs
    assert kwargs.get("dtype", kwargs.get("torch_dtype")) is torch.float32
    fake_model.to.assert_called_once_with("cuda")


def test_load_model_raises_when_the_requested_dtype_is_not_honored():
    # Newer transformers deprecate `torch_dtype=` in favour of `dtype=`; a
    # version that ignored the argument would silently load the checkpoint's
    # own dtype and invalidate the run's recorded config (Constraint #4).
    fake_tokenizer = MagicMock(pad_token="<pad>", eos_token="<eos>")
    fake_model = _fake_loaded_model(torch.float16)

    with patch.object(
        generation.AutoTokenizer, "from_pretrained", return_value=fake_tokenizer
    ), patch.object(
        generation.AutoModelForCausalLM, "from_pretrained", return_value=fake_model
    ):
        with pytest.raises(RuntimeError, match="not the requested"):
            generation.load_model(
                "Salesforce/codegen-350M-mono", device="cpu", dtype="float32"
            )


def test_load_model_falls_back_to_torch_dtype_on_older_transformers():
    # Older transformers do not accept `dtype=`; the loader must retry with
    # `torch_dtype=` rather than let the TypeError kill the run.
    fake_tokenizer = MagicMock(pad_token="<pad>", eos_token="<eos>")
    fake_model = _fake_loaded_model(torch.float32)
    calls: list[dict] = []

    def fake_from_pretrained(name, **kwargs):
        calls.append(kwargs)
        if "dtype" in kwargs:
            raise TypeError("unexpected keyword argument 'dtype'")
        return fake_model

    with patch.object(
        generation.AutoTokenizer, "from_pretrained", return_value=fake_tokenizer
    ), patch.object(
        generation.AutoModelForCausalLM,
        "from_pretrained",
        side_effect=fake_from_pretrained,
    ):
        generation.load_model(
            "Salesforce/codegen-350M-mono", device="cpu", dtype="float32"
        )

    assert [set(c) for c in calls] == [{"dtype"}, {"torch_dtype"}]
    assert calls[-1]["torch_dtype"] is torch.float32


# --- device_map (2026-08-31, the >=7B capability rung) --------------------


def test_load_model_default_device_map_keeps_the_pre_week7_load_path():
    # Constraint #4: every stored run must reproduce. The default must not
    # merely *behave* like the old path — it must BE it, i.e. the device_map
    # keyword never reaches from_pretrained, since passing even device_map=None
    # routes newer transformers through accelerate's dispatch.
    fake_tokenizer = MagicMock(pad_token="<pad>", eos_token="<eos>")
    fake_model = _fake_loaded_model(torch.float32)

    with patch.object(
        generation.AutoTokenizer, "from_pretrained", return_value=fake_tokenizer
    ), patch.object(
        generation.AutoModelForCausalLM, "from_pretrained", return_value=fake_model
    ) as model_load:
        generation.load_model(
            "Salesforce/codegen-350M-mono", device="cpu", dtype="float32"
        )

    assert "device_map" not in model_load.call_args.kwargs
    fake_model.to.assert_called_once_with("cpu")


def test_load_model_with_device_map_forwards_it_and_skips_the_to_call():
    # `.to()` on an accelerate-dispatched model raises outright, so the
    # single-device move must be skipped — not merely redundant, actively wrong.
    fake_tokenizer = MagicMock(pad_token="<pad>", eos_token="<eos>")
    fake_model = _fake_loaded_model(torch.float16)
    fake_model.hf_device_map = {"model.layers.0": 0, "lm_head": "cpu"}

    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        generation.AutoTokenizer, "from_pretrained", return_value=fake_tokenizer
    ), patch.object(
        generation.AutoModelForCausalLM, "from_pretrained", return_value=fake_model
    ) as model_load:
        generation.load_model(
            "Qwen/Qwen2.5-Coder-7B", device="cuda", dtype="float16", device_map="auto"
        )

    assert model_load.call_args.kwargs["device_map"] == "auto"
    fake_model.to.assert_not_called()
    fake_model.eval.assert_called_once()


def test_load_model_rejects_unknown_device_map():
    with pytest.raises(ValueError, match="device_map must be one of"):
        generation.load_model("Salesforce/codegen-350M-mono", device_map="everywhere")


def test_load_model_refuses_device_map_without_cuda():
    # Observed, not hypothesized: on an Apple machine device_map="auto" with
    # device="cpu" loaded onto MPS (hf_device_map == {"": "mps"}). accelerate
    # never consults the device argument, so the run would have been recorded
    # as device_resolved="cpu" while Metal kernels drew different tokens.
    with patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(RuntimeError, match="requires CUDA"):
            generation.load_model(
                "Salesforce/codegen-350M-mono", device="cpu", device_map="auto"
            )


def test_input_device_uses_the_embedding_layer_when_dispatched():
    # Parameter order does not track accelerate's dispatch plan: the first
    # parameter can be offloaded to CPU (or sit on `meta`) while the embedding
    # that consumes input_ids is on the GPU. Sending inputs to the first
    # parameter's device is the bug this guards.
    model = MagicMock()
    model.hf_device_map = {"model.embed_tokens": 0, "model.layers.31": "cpu"}
    model.parameters.return_value = iter([torch.zeros(1, device="meta")])
    embeddings = MagicMock()
    embeddings.parameters.return_value = iter([torch.zeros(1)])
    model.get_input_embeddings.return_value = embeddings

    assert generation.input_device(model).type == "cpu"


def test_input_device_falls_back_to_the_first_parameter_when_not_dispatched():
    model = MagicMock(spec=["parameters"])
    model.parameters.return_value = iter([torch.zeros(1)])

    assert generation.input_device(model).type == "cpu"


def test_multi_sample_generation_calls_generate_code_n_times_with_distinct_seeds():
    seeds_seen = []

    with patch.object(
        generation, "generate_code", side_effect=["a", "b", "c"]
    ) as fake_generate, patch.object(
        generation, "set_all_seeds", side_effect=seeds_seen.append
    ):
        result = generation.multi_sample_generation(
            "prompt", tokenizer=object(), model=object(), n=3, temperature=0.7, seed=42
        )

    assert result == ["a", "b", "c"]
    assert seeds_seen == [42, 43, 44]
    assert fake_generate.call_count == 3


def test_generate_code_uses_chat_template_when_requested():
    tokenizer = MagicMock()
    encoded = MagicMock()
    encoded.__getitem__.side_effect = lambda key: {"input_ids": torch.tensor([[1, 2]])}[
        key
    ]
    encoded.to.return_value = encoded
    tokenizer.apply_chat_template.return_value = encoded
    tokenizer.pad_token_id = 0
    tokenizer.decode.return_value = "def fixed():\n    return True\n"

    model = MagicMock()
    parameter = MagicMock()
    parameter.device = "cpu"
    model.parameters.return_value = iter([parameter])
    model.generate.return_value = torch.tensor([[1, 2, 3]])

    result = generation.generate_code(
        "repair this",
        tokenizer,
        model,
        temperature=0.0,
        use_chat_template=True,
    )

    assert result.startswith("def fixed")
    tokenizer.apply_chat_template.assert_called_once()
    tokenizer.assert_not_called()


@pytest.mark.slow
def test_generate_code_real_model_strips_prompt():
    tokenizer, model = generation.load_model("Salesforce/codegen-350M-mono")
    prompt = 'def add(a, b):\n    """Return the sum of a and b."""\n'
    completion = generation.generate_code(
        prompt, tokenizer, model, temperature=0.2, max_new_tokens=16
    )
    assert isinstance(completion, str)
    assert not completion.startswith(prompt)


@pytest.mark.slow
def test_multi_sample_generation_real_model_returns_n_samples():
    tokenizer, model = generation.load_model("Salesforce/codegen-350M-mono")
    prompt = 'def add(a, b):\n    """Return the sum of a and b."""\n'
    samples = generation.multi_sample_generation(
        prompt, tokenizer, model, n=2, temperature=0.7, seed=0
    )
    assert len(samples) == 2
    assert all(isinstance(s, str) for s in samples)


def test_early_stop_criteria_waits_for_a_stop_sequence():
    tokenizer = MagicMock()
    criteria = generation._StopSequenceCriteria(tokenizer, prompt_len=3)

    tokenizer.decode.return_value = "    return x + 1\n"
    assert criteria(torch.zeros((1, 8), dtype=torch.long), None) is False

    tokenizer.decode.return_value = "    return x + 1\n\ndef other():\n"
    assert criteria(torch.zeros((1, 12), dtype=torch.long), None) is True


def test_early_stop_criteria_requires_every_sequence_in_the_batch():
    tokenizer = MagicMock()
    criteria = generation._StopSequenceCriteria(tokenizer, prompt_len=0)
    # One row has a stop sequence, the other does not -> must not stop.
    tokenizer.decode.side_effect = ["\ndef g():\n", "    return 1\n"]
    assert criteria(torch.zeros((2, 5), dtype=torch.long), None) is False


def test_generate_code_passes_stopping_criteria_only_when_early_stop():
    encoded = MagicMock()
    encoded.__getitem__.side_effect = lambda key: {
        "input_ids": torch.zeros((1, 4), dtype=torch.long)
    }[key]
    encoded.to.return_value = encoded
    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = encoded
    fake_tokenizer.decode.return_value = "    return 1\n"
    fake_tokenizer.pad_token_id = 0

    fake_model = MagicMock()
    fake_model.generate.return_value = torch.zeros((1, 8), dtype=torch.long)
    parameter = MagicMock()
    parameter.device = "cpu"

    fake_model.parameters.return_value = iter([parameter])
    generation.generate_code("p", fake_tokenizer, fake_model, early_stop=False)
    assert fake_model.generate.call_args.kwargs["stopping_criteria"] is None

    fake_model.parameters.return_value = iter([parameter])
    generation.generate_code("p", fake_tokenizer, fake_model, early_stop=True)
    criteria = fake_model.generate.call_args.kwargs["stopping_criteria"]
    assert criteria is not None and len(criteria) == 1


@pytest.mark.slow
def test_early_stop_produces_identical_output_to_full_decoding():
    """The optimization's whole justification: same string, less compute.

    Not mocked, because the claim is about real sampled tokens.
    """
    from src.utils import load_dataset_split, set_all_seeds

    tokenizer, model = generation.load_model("Salesforce/codegen-350M-mono")
    for problem in load_dataset_split("humaneval")[:3]:
        for seed in (42, 43):
            set_all_seeds(seed)
            full = generation.generate_code(
                problem["prompt"], tokenizer, model, temperature=0.7, early_stop=False
            )
            set_all_seeds(seed)
            early = generation.generate_code(
                problem["prompt"], tokenizer, model, temperature=0.7, early_stop=True
            )
            assert early == full, f"{problem['task_id']} seed={seed}"

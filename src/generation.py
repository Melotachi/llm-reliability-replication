"""LLM loading and multi-sample code generation.

Jointly owned by Student A and Student B (see CLAUDE.md Section 6.1).
Backend is HuggingFace Transformers only — no paid APIs (CLAUDE.md Constraint #1).
Default model: Salesforce/codegen-350M-mono. Never hardcode the model name in
callers; it must always come from a CLI argument or config.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

from src.utils import get_logger, set_all_seeds

logger = get_logger(__name__)

# Chen et al. (HumanEval/Codex paper, Section 3.2) truncate completions at
# the first of these sequences to isolate just the target function. Without
# this, a causal LM keeps generating past the function's logical end (into
# a second function, a comment, etc.), and max_new_tokens then frequently
# cuts that trailing, irrelevant text off mid-syntax — producing a
# SyntaxError on exec() even when the target function itself is correct.
_STOP_SEQUENCES = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")


def _truncate_at_stop_sequence(text: str) -> str:
    """Cut a completion at the first HumanEval-style stop sequence, if any."""
    cut_points = [
        idx for idx in (text.find(seq) for seq in _STOP_SEQUENCES) if idx != -1
    ]
    return text[: min(cut_points)] if cut_points else text


class _StopSequenceCriteria(StoppingCriteria):
    """Halt generation once a stop sequence is present in the completion.

    Purely a speed optimization, and an output-preserving one. Completions are
    truncated at the *first* stop sequence anyway, and continuing to decode can
    only append text after that point, so the truncated result is unchanged --
    while the stored runs show completions average 56 tokens against a
    max_new_tokens of 256, i.e. ~78% of every generation call was being spent
    on tokens that get thrown away.

    The equality this relies on is verified against real generations in
    tests/test_generation.py rather than assumed.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, prompt_len: int) -> None:
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        """Return True once every sequence in the batch has a stop sequence."""
        for sequence in input_ids:
            completion = self._tokenizer.decode(
                sequence[self._prompt_len :], skip_special_tokens=True
            )
            if not any(seq in completion for seq in _STOP_SEQUENCES):
                return False
        return True


#: Accepted values for load_model()'s keyword-only overrides. "auto" is the
#: pre-Week-7 behavior and stays the default everywhere, so every run recorded
#: before these flags existed reproduces unchanged (CLAUDE.md Constraint #4).
DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
DTYPE_CHOICES = ("auto", "float32", "float16", "bfloat16")

#: Accepted values for load_model()'s `device_map`. "none" is the pre-Week-7
#: single-device path (a plain `model.to(device)`) and stays the default, so
#: every stored run reproduces unchanged. Anything else is handed to
#: accelerate via from_pretrained(device_map=...), which loads the checkpoint
#: shard-by-shard straight to its target and spills whatever does not fit to
#: CPU instead of raising OutOfMemoryError.
#:
#: This matters only at the top of the capability ladder. Qwen2.5-Coder-7B is
#: ~15.2 GB in float16 against a P100's ~15.9 GB usable, and the single-device
#: path materializes the weights in CPU RAM and *then* allocates the same
#: bytes again on the GPU, so allocator fragmentation sinks it well before
#: the arithmetic says it should fit.
#:
#: Offloading is numerically neutral: accelerate moves each weight onto the
#: GPU before its matmul, so an offloaded forward pass produces the same
#: values as a fully resident one, only slower. There is therefore no
#: comparability cost to enabling it — unlike quantization, which changes the
#: numerics and is deliberately not offered here.
DEVICE_MAP_CHOICES = ("none", "auto", "balanced", "balanced_low_0", "sequential")

_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_device(device: str) -> str:
    """Turn a DEVICE_CHOICES value into a concrete torch device string."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    # Deliberately NOT falling through to MPS: every stored run of this
    # project was generated on CPU, and Apple's Metal kernels produce
    # different sampling draws for the same seed. Opt in with --device mps.
    return "cpu"


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    """Turn a DTYPE_CHOICES value into a torch dtype for the given device."""
    if dtype != "auto":
        return _DTYPE_BY_NAME[dtype]
    # float16 matmul kernels are frequently unsupported or slow on CPU.
    return torch.float16 if device == "cuda" else torch.float32


def resolve_runtime(
    device: str = "auto", dtype: str = "auto"
) -> tuple[str, torch.dtype]:
    """Resolve a (device, dtype) request into what load_model would actually use.

    Public because callers need to *record* the resolved pair without holding a
    model: "auto" alone does not identify a run, and both device and dtype
    change the sampled tokens for a fixed seed (CLAUDE.md Constraint #4).

    Args:
        device: One of DEVICE_CHOICES.
        dtype: One of DTYPE_CHOICES.
    Returns:
        (concrete device string, concrete torch dtype).
    Raises:
        ValueError: If either argument is outside its choices tuple.
    """
    if device not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES}, got {device!r}")
    if dtype not in DTYPE_CHOICES:
        raise ValueError(f"dtype must be one of {DTYPE_CHOICES}, got {dtype!r}")
    resolved_device = _resolve_device(device)
    return resolved_device, _resolve_dtype(dtype, resolved_device)


def _load_with_dtype(
    model_name: str, dtype: torch.dtype, device_map: str | None = None
) -> PreTrainedModel:
    """Load a checkpoint at `dtype`, whichever keyword this transformers wants.

    `torch_dtype=` is the historical name and is deprecated in favour of
    `dtype=`; the changeover happened between the version the stored runs were
    generated with (4.50) and the ones on hosted GPU images. Passing the wrong
    one silently loads the checkpoint's own dtype instead — see load_model()'s
    verification for why that invalidates a run — so try the modern spelling
    first and fall back, rather than pinning either.

    `device_map` is forwarded only when set, so the keyword never appears in
    the call for the default path and the pre-Week-7 behavior is unchanged.
    """
    extra = {"device_map": device_map} if device_map is not None else {}
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **extra)
    except TypeError:
        logger.debug("transformers rejected dtype=; retrying with torch_dtype=")
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, **extra
        )


def input_device(model: PreTrainedModel) -> torch.device:
    """Return the device that `model`'s input tensors must be placed on.

    The obvious `next(model.parameters()).device` is wrong for an
    accelerate-dispatched model: parameter order does not track the dispatch
    plan, so the first parameter can sit on CPU (offloaded) or on `meta` while
    the embedding layer that actually consumes `input_ids` lives on the GPU.
    Sending inputs to a `meta` device yields tensors with no storage.

    Args:
        model: A loaded causal LM, dispatched or not.
    Returns:
        The device holding the input embeddings, falling back to the first
        parameter's device for models without an embedding lookup.
    """
    if getattr(model, "hf_device_map", None):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return next(embeddings.parameters()).device
    return next(model.parameters()).device


def load_model(
    model_name: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    device_map: str = "none",
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load tokenizer and model from HuggingFace Hub.

    Caches to ~/.cache/huggingface by default. Under the default
    ``device="auto"``/``dtype="auto"``/``device_map="none"`` this picks CUDA +
    float16 when a GPU is available and CPU + float32 otherwise, on a single
    device — bit-for-bit the behavior every run before Week 7 used.

    Args:
        model_name: HuggingFace model identifier.
                    Default: "Salesforce/codegen-350M-mono"
        device: One of DEVICE_CHOICES. Keyword-only. "auto" never selects
            MPS — see _resolve_device() for why. Ignored for *placement* when
            device_map is set, since accelerate then owns the layout.
        dtype: One of DTYPE_CHOICES. Keyword-only. Pin this to "float32" when
            a run must stay numerically comparable to the CPU-generated runs;
            "float16" roughly halves GPU generation time but changes the
            sampled tokens.
        device_map: One of DEVICE_MAP_CHOICES. Keyword-only. "none" keeps the
            single-device path. Any other value hands placement to accelerate,
            which spills layers that do not fit to CPU rather than raising —
            see DEVICE_MAP_CHOICES for why that costs nothing scientifically.
    Returns:
        (tokenizer, model) tuple ready for inference.
    Raises:
        ValueError: If device, dtype or device_map is outside its choices.
        ImportError: If device_map is set but accelerate is not installed.
    """
    if device_map not in DEVICE_MAP_CHOICES:
        raise ValueError(
            f"device_map must be one of {DEVICE_MAP_CHOICES}, got {device_map!r}"
        )
    device, dtype = resolve_runtime(device, dtype)
    resolved_map = None if device_map == "none" else device_map

    # accelerate picks placement on its own and does NOT consult the `device`
    # argument, so on an Apple machine device_map="auto" silently loads onto
    # MPS even when "cpu" was requested — verified: hf_device_map came back
    # {"": "mps"}. That breaks two of this project's rules at once:
    # _resolve_device() deliberately never selects MPS because Metal kernels
    # change the sampled draws at a fixed seed, and the results config would
    # then record device_resolved="cpu" for a run that was not on the CPU.
    # The sharding path exists only to fit large weights across CUDA devices
    # and spill to host RAM, so restricting it to CUDA costs nothing real.
    if resolved_map is not None and not torch.cuda.is_available():
        raise RuntimeError(
            f"device_map={device_map!r} requires CUDA: accelerate chooses "
            "placement itself and would select MPS or CPU regardless of "
            f"device={device!r}, silently invalidating the recorded config. "
            'Use device_map="none" on non-CUDA machines.'
        )

    logger.info(
        "Loading %s on %s (dtype=%s, device_map=%s)",
        model_name,
        device,
        dtype,
        resolved_map or "none",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_with_dtype(model_name, dtype, resolved_map)
    if resolved_map is None:
        model.to(device)
    else:
        # `.to()` on an accelerate-dispatched model raises; accelerate has
        # already placed every module. Log the plan instead: it is the only
        # record of whether the run was GPU-resident or partly offloaded,
        # which is what explains the wall clock afterwards.
        logger.info("accelerate device_map: %s", getattr(model, "hf_device_map", {}))
    model.eval()

    # Verify rather than assume. `torch_dtype=` is deprecated in favour of
    # `dtype=` in newer transformers, and every checkpoint this project uses
    # declares a non-float32 `torch_dtype` in its own config (codegen-350M
    # float16; deepseek-1.3b and both Qwen coders bfloat16). A version that
    # dropped the argument and fell back to the config value would load a
    # *different* dtype than the one recorded in the results JSON, silently
    # invalidating the run — CLAUDE.md Constraint #4 turns on dtype, since it
    # changes the sampled tokens at a fixed seed. Fail loudly instead.
    loaded_dtype = next(model.parameters()).dtype
    if loaded_dtype != dtype:
        raise RuntimeError(
            f"{model_name} loaded as {loaded_dtype}, not the requested {dtype}. "
            "The installed transformers version is not honouring the dtype "
            "argument; results from this environment are not comparable to "
            "runs recorded with the requested dtype."
        )
    logger.info("Loaded %s (verified dtype=%s)", model_name, loaded_dtype)
    return tokenizer, model


def generate_code(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    use_chat_template: bool = False,
    early_stop: bool = True,
) -> str:
    """Generate a single code completion for the given prompt.

    Args:
        prompt: The natural language + signature prompt from the dataset.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace causal LM model.
        temperature: Sampling temperature (higher = more diverse). Values
            <= 0 fall back to greedy decoding (HF errors on temperature=0
            with do_sample=True).
        max_new_tokens: Max tokens to generate beyond the prompt.
        use_chat_template: Wrap the prompt as a user message for an
            instruction-tuned/chat model.
        early_stop: Stop as soon as a stop sequence appears instead of always
            decoding `max_new_tokens` and truncating afterwards. Output-
            identical (see _StopSequenceCriteria) and roughly 4x faster on this
            project's completion-length distribution. Ignored under
            use_chat_template, where no stop-sequence truncation applies.
    Returns:
        Generated code as a string (prompt portion stripped).
    """
    device = input_device(model)
    if use_chat_template:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.0
    stopping_criteria = None
    if early_stop and not use_chat_template:
        stopping_criteria = StoppingCriteriaList(
            [_StopSequenceCriteria(tokenizer, prompt_len)]
        )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
        )

    generated_ids = output_ids[0][prompt_len:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return decoded if use_chat_template else _truncate_at_stop_sequence(decoded)


def multi_sample_generation(
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    n: int = 5,
    temperature: float = 0.7,
    seed: int = 42,
    max_new_tokens: int = 256,
) -> list[str]:
    """Generate N independent code samples for a single prompt.

    Each sample is generated independently (separate forward passes). Each
    sample uses seed+i so the batch is reproducible but not degenerate
    (all-identical) under sampling.

    Args:
        prompt: Problem prompt.
        tokenizer, model: From load_model().
        n: Number of samples to generate.
        temperature: Sampling temperature.
        seed: Base random seed (each sample uses seed+i for independence).
        max_new_tokens: Max tokens per sample. The 256 default is what every
            stored HumanEval/MBPP run used; BigCodeBench solutions are longer
            (canonical bodies mean 466 / p90 787 characters), so a cell there
            must raise it and report the truncation rate, since truncated code
            fails to parse and unparseable samples count as maximally distant
            -- which would inflate the diversity metrics as a harness artefact.
    Returns:
        List of N generated code strings.
    """
    outputs = []
    for i in range(n):
        set_all_seeds(seed + i)
        outputs.append(
            generate_code(
                prompt,
                tokenizer,
                model,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        )
    return outputs

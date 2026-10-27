"""Tests for src/uncertainty.py.

Style: small hand-computable examples plus edge cases (all-identical,
all-unparseable, empty), per the Week 1 report's testing pattern. Expected
values are derived in comments so they stay checkable by hand.

Week 3: codebert_cosine_diversity unit tests mock the embedding step (per
the CLAUDE.md FAQ, no real model loads outside @pytest.mark.slow); the one
real-encoder integration test is marked slow.

Week 4: self-consistency functions follow the same hand-computable-pair
style; UncertaintyPredictor / evaluate_predictor_cv tests use small
synthetic datasets (no real uncertainty data needed to test the ML
mechanics) and one test spies on StratifiedGroupKFold to directly verify
no group's rows ever split across train/test within a fold.

Week 5: compute_calibration tests use hand-constructed (y_true, y_prob)
pairs where the ECE/Brier values are computable by hand (perfectly
calibrated, uniformly overconfident, degenerate single-value inputs).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from sklearn.model_selection import StratifiedGroupKFold

from src import uncertainty
from src.uncertainty import (
    UncertaintyPredictor,
    _normalized_levenshtein,
    ast_node_diversity,
    ast_node_self_consistency,
    codebert_bertscore_diversity,
    codebert_cosine_diversity,
    codebert_cosine_self_consistency,
    compute_calibration,
    compute_self_consistency,
    compute_uncertainty,
    compute_uncertainty_prompt_aware,
    evaluate_predictor_cv,
    execution_behavior_diversity,
    execution_behavior_diversity_nocrash,
    execution_behavior_self_consistency,
    execution_behavior_self_consistency_nocrash,
    execution_crash_rate,
    lexical_diversity,
    pairwise_distance,
    parse_failure_rate,
    prepare_metric_inputs,
    prompt_confound_covariates,
    split_prompt_completion,
    strip_prompt_docstring,
    token_cosine_diversity,
    token_cosine_self_consistency,
    token_jaccard_diversity,
    token_jaccard_self_consistency,
    tree_edit_distance_diversity,
)

# ---------------------------------------------------------------------------
# lexical_diversity (Week 2: whitespace-insensitive lexeme sequences)
# ---------------------------------------------------------------------------


def test_lexical_diversity_all_unique():
    assert lexical_diversity(["a", "b", "c"]) == 1.0


def test_lexical_diversity_all_identical():
    assert lexical_diversity(["a", "a", "a", "a"]) == 1 / 4


def test_lexical_diversity_empty():
    assert lexical_diversity([]) == 0.0


def test_lexical_diversity_collapses_whitespace_variants():
    # Week 2 improvement: all three are the lexeme sequence (x, =, 1), so
    # they count as ONE unique output. Week 1's exact-string version
    # returned 1.0 here — that saturation is what motivated the change.
    assert lexical_diversity(["x = 1", "x  =  1", "x=1"]) == 1 / 3


def test_lexical_diversity_keeps_operator_differences():
    # Punctuation is kept as lexemes: (a, +, b) vs (a, -, b) are distinct.
    assert lexical_diversity(["a+b", "a-b"]) == 1.0


def test_lexical_diversity_blank_lines_do_not_count():
    assert (
        lexical_diversity(["def f():\n    return 1", "def f():\n\n    return 1"])
        == 1 / 2
    )


# ---------------------------------------------------------------------------
# token_jaccard_diversity
# ---------------------------------------------------------------------------


def test_token_jaccard_diversity_identical_outputs():
    assert token_jaccard_diversity(["def f(): return 1"] * 3) == 0.0


def test_token_jaccard_diversity_disjoint_outputs():
    assert token_jaccard_diversity(["a b", "c d"]) == 1.0


def test_token_jaccard_diversity_hand_computed_pair():
    # Token sets {a, b} and {b, c}: |∩| = 1, |∪| = 3 → distance 1 - 1/3 = 2/3.
    assert token_jaccard_diversity(["a b", "b c"]) == pytest.approx(2 / 3)


def test_token_jaccard_diversity_ignores_punctuation():
    # Section 6.2 contract: punctuation is a separator, so both outputs
    # have token set {f, x} and the distance is 0.
    assert token_jaccard_diversity(["f(x)", "f x"]) == 0.0


def test_token_jaccard_diversity_fewer_than_two_outputs():
    assert token_jaccard_diversity([]) == 0.0
    assert token_jaccard_diversity(["only one"]) == 0.0


def test_token_jaccard_diversity_empty_strings_are_identical():
    # Two outputs with no tokens at all count as identical, not distant.
    assert token_jaccard_diversity(["", "   "]) == 0.0


def test_token_jaccard_diversity_empty_vs_nonempty_is_maximal():
    assert token_jaccard_diversity(["", "a"]) == 1.0


# ---------------------------------------------------------------------------
# token_cosine_diversity (Week 2 addition: cosine over token count vectors)
# ---------------------------------------------------------------------------


def test_token_cosine_diversity_identical_outputs():
    # sqrt(x)*sqrt(x) is not always exactly x in floating point, so this
    # goes through pytest.approx rather than exact equality (unlike
    # Jaccard, which stays in exact integer/rational arithmetic).
    assert token_cosine_diversity(["def f(): return 1"] * 3) == pytest.approx(0.0)


def test_token_cosine_diversity_disjoint_outputs():
    # Numerator (dot product) is exactly 0 here, so no sqrt rounding
    # enters the result — exact equality is safe.
    assert token_cosine_diversity(["a b", "c d"]) == 1.0


def test_token_cosine_diversity_hand_computed_pair():
    # Count vectors {a:1, b:1} and {b:1, c:1}: dot = 1 (shared b),
    # norms = sqrt(2) each -> cos = 1/2 -> distance = 0.5.
    assert token_cosine_diversity(["a b", "b c"]) == pytest.approx(0.5)


def test_token_cosine_diversity_ignores_punctuation():
    # Same Section 6.2 token contract as token_jaccard_diversity.
    assert token_cosine_diversity(["f(x)", "f x"]) == pytest.approx(0.0)


def test_token_cosine_diversity_fewer_than_two_outputs():
    assert token_cosine_diversity([]) == 0.0
    assert token_cosine_diversity(["only one"]) == 0.0


def test_token_cosine_diversity_empty_strings_are_identical():
    assert token_cosine_diversity(["", "   "]) == 0.0


def test_token_cosine_diversity_empty_vs_nonempty_is_maximal():
    assert token_cosine_diversity(["", "a"]) == 1.0


def test_token_cosine_diversity_sees_multiplicity_jaccard_misses():
    # The headline motivation: identical token SETS {a, b} but different
    # counts (a x3, b x1 vs. a x1, b x3) -> Jaccard distance 0.0, cosine
    # distance > 0. dot = 3*1 + 1*3 = 6, norms = sqrt(10) each ->
    # cos = 6/10 -> distance = 0.4.
    outputs = ["a a a b", "a b b b"]
    assert token_jaccard_diversity(outputs) == 0.0
    assert token_cosine_diversity(outputs) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# ast_node_diversity
# ---------------------------------------------------------------------------


def test_ast_node_diversity_identical_outputs():
    assert ast_node_diversity(["def f():\n    return 1\n"] * 3) == 0.0


def test_ast_node_diversity_ignores_names_and_constants():
    # Same structure, different identifiers and literals → node-type
    # sequences are identical → distance 0. This is the property that
    # separates the structural metric from the lexical ones.
    a = "def f(a):\n    return a + 1\n"
    b = "def g(x):\n    return x + 2\n"
    assert ast_node_diversity([a, b]) == 0.0


def test_ast_node_diversity_structural_difference_hand_computed():
    # "x = 1"            → (Module, Assign, Name, Store, Constant)
    # "def f(): return 1" → (Module, FunctionDef, arguments, Return, Constant)
    # Equal length 5, first and last symbols match, middle 3 all differ
    # → Levenshtein 3 → normalized 3/5.
    assert ast_node_diversity(["x = 1\n", "def f():\n    return 1\n"]) == pytest.approx(
        0.6
    )


def test_ast_node_diversity_unparseable_is_maximally_distant():
    assert ast_node_diversity(["def f(:\n", "x = 1\n"]) == 1.0


def test_ast_node_diversity_two_unparseable_samples():
    # Documented design decision: no structure to compare → maximal.
    assert ast_node_diversity(["def f(:", "def g(:"]) == 1.0


def test_ast_node_diversity_mixed_hand_computed():
    # Pairs: (valid, valid identical) = 0.0, (valid, broken) = 1.0 twice
    # → mean = 2/3.
    outputs = ["x = 1", "x = 1", "def f(:"]
    assert ast_node_diversity(outputs) == pytest.approx(2 / 3)


def test_ast_node_diversity_fewer_than_two_outputs():
    assert ast_node_diversity([]) == 0.0
    assert ast_node_diversity(["x = 1"]) == 0.0


def test_ast_node_diversity_flattening_limitation_documented():
    # Known Week 2 limitation, kept visible on purpose: pre-order type
    # sequences discard nesting, so moving b() out of the if-body leaves
    # the sequence unchanged and the distance at 0.0. Week 3's tree edit
    # distance is the planned fix for exactly this blind spot.
    inside = "if x:\n    a()\n    b()\n"
    outside = "if x:\n    a()\nb()\n"
    assert ast_node_diversity([inside, outside]) == 0.0


# ---------------------------------------------------------------------------
# tree_edit_distance_diversity (Week 3)
# ---------------------------------------------------------------------------


def test_tree_edit_distance_identical_outputs():
    assert tree_edit_distance_diversity(["def f():\n    return 1\n"] * 3) == 0.0


def test_tree_edit_distance_ignores_names_and_constants():
    # Same structure, different identifiers/literals → same node-type tree
    # → distance 0, exactly like ast_node_diversity.
    a = "def f(a):\n    return a + 1\n"
    b = "def g(x):\n    return x + 2\n"
    assert tree_edit_distance_diversity([a, b]) == 0.0


def test_tree_edit_distance_hand_computed_pair():
    # "x = 1"             → Module(Assign(Name(Store), Constant)), 5 nodes
    # "def f(): return 1" → Module(FunctionDef(arguments, Return(Constant))), 5 nodes
    # Optimal mapping: keep Module and Constant (0), relabel
    # Assign→FunctionDef (1) and Name→arguments (1), delete Store (1),
    # insert Return (1) → TED 4, normalized by 5 + 5 → 0.4.
    assert tree_edit_distance_diversity(
        ["x = 1\n", "def f():\n    return 1\n"]
    ) == pytest.approx(0.4)


def test_tree_edit_distance_closes_flattening_blind_spot():
    # The Week 3 headline. This is the exact pair the Week 2 report
    # documents as ast_node_diversity's blind spot: nesting-only difference,
    # identical pre-order type sequences → 0.0. True TED sees it: delete
    # the If (its children hoist to Module), re-insert it adopting only the
    # first two — 2 operations over 12 + 12 nodes → 1/12.
    inside = "if x:\n    a()\n    b()\n"
    outside = "if x:\n    a()\nb()\n"
    assert ast_node_diversity([inside, outside]) == 0.0  # still blind
    assert tree_edit_distance_diversity([inside, outside]) == pytest.approx(1 / 12)


def test_tree_edit_distance_unparseable_is_maximally_distant():
    assert tree_edit_distance_diversity(["def f(:\n", "x = 1\n"]) == 1.0


def test_tree_edit_distance_two_unparseable_samples():
    # Same documented design decision as ast_node_diversity: no structure
    # to compare → maximal.
    assert tree_edit_distance_diversity(["def f(:", "def g(:"]) == 1.0


def test_tree_edit_distance_mixed_hand_computed():
    # Pairs: (valid, valid identical) = 0.0, (valid, broken) = 1.0 twice
    # → mean = 2/3.
    outputs = ["x = 1", "x = 1", "def f(:"]
    assert tree_edit_distance_diversity(outputs) == pytest.approx(2 / 3)


def test_tree_edit_distance_fewer_than_two_outputs():
    assert tree_edit_distance_diversity([]) == 0.0
    assert tree_edit_distance_diversity(["x = 1"]) == 0.0


def test_tree_edit_distance_valid_pair_strictly_below_one():
    # Both roots are Module, so any parseable pair maps them for free and
    # stays strictly below the 1.0 reserved for parse failures.
    outputs = [
        "x = 1\n",
        "class A:\n    def m(self):\n        while True:\n            yield 1\n",
    ]
    value = tree_edit_distance_diversity(outputs)
    assert 0.0 < value < 1.0


# ---------------------------------------------------------------------------
# codebert_cosine_diversity (Week 3) — embedding step mocked in unit tests
# ---------------------------------------------------------------------------


def test_codebert_cosine_identical_embeddings_give_zero():
    fake = np.array([[3.0, 0.0], [3.0, 0.0]])
    with patch.object(uncertainty, "_embed_outputs", return_value=fake):
        assert codebert_cosine_diversity(["a", "b"]) == pytest.approx(0.0)


def test_codebert_cosine_is_scale_invariant():
    # Cosine depends on direction only: (2, 0) vs (1, 0) are identical.
    fake = np.array([[2.0, 0.0], [1.0, 0.0]])
    with patch.object(uncertainty, "_embed_outputs", return_value=fake):
        assert codebert_cosine_diversity(["a", "b"]) == pytest.approx(0.0)


def test_codebert_cosine_orthogonal_embeddings_give_one():
    fake = np.array([[1.0, 0.0], [0.0, 1.0]])
    with patch.object(uncertainty, "_embed_outputs", return_value=fake):
        assert codebert_cosine_diversity(["a", "b"]) == pytest.approx(1.0)


def test_codebert_cosine_opposite_embeddings_give_two():
    # Documents the theoretical [0, 2] range of cosine distance.
    fake = np.array([[1.0, 0.0], [-1.0, 0.0]])
    with patch.object(uncertainty, "_embed_outputs", return_value=fake):
        assert codebert_cosine_diversity(["a", "b"]) == pytest.approx(2.0)


def test_codebert_cosine_mean_over_pairs_hand_computed():
    # Pairs: (e1,e2) = 1.0, (e1,e3) = 0.0, (e2,e3) = 1.0 → mean = 2/3.
    fake = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    with patch.object(uncertainty, "_embed_outputs", return_value=fake):
        assert codebert_cosine_diversity(["a", "b", "c"]) == pytest.approx(2 / 3)


def test_codebert_cosine_fewer_than_two_outputs_skips_model():
    # < 2 outputs returns 0.0 without touching the encoder at all.
    with patch.object(uncertainty, "_embed_outputs") as fake_embed:
        assert codebert_cosine_diversity([]) == 0.0
        assert codebert_cosine_diversity(["only one"]) == 0.0
    fake_embed.assert_not_called()


class _FakeBatch(dict):
    """Minimal stand-in for a HuggingFace BatchEncoding: dict + .to(device)."""

    def to(self, device):
        return self


class _FakeEncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeEncoder:
    """Counts forward-pass calls so the embedding cache's hit/miss behavior
    is directly observable, without needing a real HuggingFace model."""

    def __init__(self, hidden_size: int = 4):
        self._param = torch.zeros(1)
        self.hidden_size = hidden_size
        self.calls = 0

    def parameters(self):
        return iter([self._param])

    def __call__(self, **kwargs):
        self.calls += 1
        input_ids = kwargs["input_ids"]
        return _FakeEncoderOutput(
            torch.ones((input_ids.shape[0], input_ids.shape[1], self.hidden_size))
        )


def _fake_tokenizer(outputs, **kwargs):
    return _FakeBatch(
        input_ids=torch.zeros((len(outputs), 3), dtype=torch.long),
        attention_mask=torch.ones((len(outputs), 3)),
    )


def test_embed_outputs_caches_repeated_calls():
    # _embed_outputs is called once by codebert_cosine_diversity and again
    # by codebert_cosine_self_consistency for the same problem's samples;
    # the cache must make the second call a no-op against the encoder.
    uncertainty._EMBEDDING_CACHE.clear()
    fake_encoder = _FakeEncoder()
    try:
        with patch.object(
            uncertainty, "_load_codebert", return_value=(_fake_tokenizer, fake_encoder)
        ):
            first = uncertainty._embed_outputs(["a", "b"], "fake-model")
            second = uncertainty._embed_outputs(["a", "b"], "fake-model")  # cache hit
            different = uncertainty._embed_outputs(
                ["a", "c"], "fake-model"
            )  # cache miss
        assert fake_encoder.calls == 2  # not 3: the repeated call was cached
        assert np.array_equal(first, second)
        assert first is second  # exact cached object, not a recomputed copy
        assert different.shape == first.shape
    finally:
        uncertainty._EMBEDDING_CACHE.clear()


def test_embed_outputs_cache_keyed_by_model_name_too():
    # Same outputs, different model_name, must not collide in the cache.
    uncertainty._EMBEDDING_CACHE.clear()
    fake_encoder = _FakeEncoder()
    try:
        with patch.object(
            uncertainty, "_load_codebert", return_value=(_fake_tokenizer, fake_encoder)
        ):
            uncertainty._embed_outputs(["a", "b"], "model-one")
            uncertainty._embed_outputs(["a", "b"], "model-two")
        assert fake_encoder.calls == 2
    finally:
        uncertainty._EMBEDDING_CACHE.clear()


@pytest.mark.slow
def test_codebert_cosine_diversity_real_encoder():
    # Real microsoft/codebert-base (~500 MB, cached after first download).
    identical = ["def f(a):\n    return a + 1\n"] * 2
    different = [
        "def f(a):\n    return a + 1\n",
        "import os\n\nfor p in os.listdir('.'):\n    print(p.upper())\n",
    ]
    same_value = codebert_cosine_diversity(identical)
    diff_value = codebert_cosine_diversity(different)
    assert same_value == pytest.approx(0.0, abs=1e-5)
    assert 0.0 < diff_value < 2.0
    assert diff_value > same_value


# ---------------------------------------------------------------------------
# codebert_bertscore_diversity (Week 3 addition) — embedding step mocked
# ---------------------------------------------------------------------------


def test_codebert_bertscore_identical_single_token_gives_zero():
    fake = [np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])]
    with patch.object(uncertainty, "_embed_tokens", return_value=fake):
        assert codebert_bertscore_diversity(["a", "b"]) == pytest.approx(0.0)


def test_codebert_bertscore_orthogonal_single_token_gives_one():
    fake = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    with patch.object(uncertainty, "_embed_tokens", return_value=fake):
        assert codebert_bertscore_diversity(["a", "b"]) == pytest.approx(1.0)


def test_codebert_bertscore_hand_computed_asymmetric_pair():
    # a has 1 token (1,0); b has 2 tokens (1,0) and (0,1).
    # precision (mean over a's tokens of max sim into b) = max(1, 0) = 1.0
    # recall    (mean over b's tokens of max sim into a) = mean(1, 0) = 0.5
    # F1 = 2*1*0.5 / 1.5 = 2/3 -> distance = 1 - 2/3 = 1/3.
    fake = [np.array([[1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])]
    with patch.object(uncertainty, "_embed_tokens", return_value=fake):
        assert codebert_bertscore_diversity(["a", "b"]) == pytest.approx(1 / 3)


def test_codebert_bertscore_no_real_tokens_is_maximal():
    # An empty-string sample has zero real tokens after special/pad
    # stripping -> no structure to compare -> maximal distance, same
    # convention as ast_node_diversity's unparseable case.
    fake = [np.zeros((0, 2)), np.array([[1.0, 0.0]])]
    with patch.object(uncertainty, "_embed_tokens", return_value=fake):
        assert codebert_bertscore_diversity(["", "b"]) == pytest.approx(1.0)


def test_codebert_bertscore_fewer_than_two_outputs_skips_model():
    with patch.object(uncertainty, "_embed_tokens") as fake_embed:
        assert codebert_bertscore_diversity([]) == 0.0
        assert codebert_bertscore_diversity(["only one"]) == 0.0
    fake_embed.assert_not_called()


@pytest.mark.slow
def test_codebert_bertscore_diversity_real_encoder():
    # Real microsoft/codebert-base (~500 MB, cached after first download).
    identical = ["def f(a):\n    return a + 1\n"] * 2
    different = [
        "def f(a):\n    return a + 1\n",
        "import os\n\nfor p in os.listdir('.'):\n    print(p.upper())\n",
    ]
    same_value = codebert_bertscore_diversity(identical)
    diff_value = codebert_bertscore_diversity(different)
    assert same_value == pytest.approx(0.0, abs=1e-4)
    assert diff_value > same_value


# ---------------------------------------------------------------------------
# _normalized_levenshtein helper
# ---------------------------------------------------------------------------


def test_normalized_levenshtein_identical():
    assert _normalized_levenshtein(("a", "b"), ("a", "b")) == 0.0


def test_normalized_levenshtein_single_substitution():
    assert _normalized_levenshtein(("a", "b"), ("a", "c")) == 0.5


def test_normalized_levenshtein_length_mismatch():
    # Two insertions to turn (a,) into (a, b, c) → 2 / max(1, 3) = 2/3.
    assert _normalized_levenshtein(("a",), ("a", "b", "c")) == pytest.approx(2 / 3)


def test_normalized_levenshtein_empty_cases():
    assert _normalized_levenshtein((), ()) == 0.0
    assert _normalized_levenshtein((), ("a",)) == 1.0
    assert _normalized_levenshtein(("a",), ()) == 1.0


# ---------------------------------------------------------------------------
# pairwise_distance (legacy Week 1 baseline, retained for comparison)
# ---------------------------------------------------------------------------


def test_pairwise_distance_matches_manual_computation():
    outputs = ["a", "abc", "ab"]  # lengths 1, 3, 2 -> pairwise |diffs| = 2, 1, 1
    assert pairwise_distance(outputs) == (2 + 1 + 1) / 3


def test_pairwise_distance_single_output_is_zero():
    assert pairwise_distance(["only one"]) == 0.0


# ---------------------------------------------------------------------------
# parse_failure_rate
# ---------------------------------------------------------------------------


def test_parse_failure_rate_mixed_validity():
    outputs = ["def f():\n    return 1\n", "def f(:\n  broken", "x = 1"]
    assert parse_failure_rate(outputs) == 1 / 3


def test_parse_failure_rate_all_valid():
    assert parse_failure_rate(["x = 1", "y = 2"]) == 0.0


# ---------------------------------------------------------------------------
# compute_uncertainty
# ---------------------------------------------------------------------------


def test_compute_uncertainty_returns_week3_default_keys():
    # Default: everything except the opt-in CodeBERT metrics (no encoder
    # download in unit tests or casual calls).
    result = compute_uncertainty(["x = 1", "x  =  1", "y = 2"])
    assert set(result.keys()) == {
        "lexical_diversity",
        "token_jaccard_diversity",
        "token_cosine_diversity",
        "ast_node_diversity",
        "tree_edit_distance_diversity",
        "pairwise_distance",
        "parse_failure_rate",
    }
    # The whitespace variants collapse under the improved lexical metric.
    assert result["lexical_diversity"] == pytest.approx(2 / 3)


def test_compute_uncertainty_include_codebert_adds_keys():
    fake_pooled = np.array([[1.0, 0.0], [0.0, 1.0]])
    fake_tokens = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    with (
        patch.object(
            uncertainty, "_embed_outputs", return_value=fake_pooled
        ) as fake_embed_outputs,
        patch.object(
            uncertainty, "_embed_tokens", return_value=fake_tokens
        ) as fake_embed_tokens,
    ):
        result = compute_uncertainty(
            ["x = 1", "y = 2"],
            include_codebert=True,
            codebert_model_name="mock/encoder",
        )
    assert result["codebert_cosine_diversity"] == pytest.approx(1.0)
    assert result["codebert_bertscore_diversity"] == pytest.approx(1.0)
    fake_embed_outputs.assert_called_once_with(["x = 1", "y = 2"], "mock/encoder")
    fake_embed_tokens.assert_called_once_with(["x = 1", "y = 2"], "mock/encoder")


def test_compute_uncertainty_values_in_range():
    outputs = [
        "def f(a):\n    return a + 1\n",
        "def f(a):\n    total = a + 1\n    return total\n",
        "def f(:\n  broken",
    ]
    result = compute_uncertainty(outputs)
    for key, value in result.items():
        if key == "pairwise_distance":  # character units, unbounded above
            assert value >= 0.0
        else:
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"
    assert result["parse_failure_rate"] == pytest.approx(1 / 3)


def test_compute_uncertainty_lexical_vs_structural_complementarity():
    # Renamed variables: lexically diverse (different identifiers) but
    # structurally identical — jaccard fires, ast stays at 0.
    outputs = ["def f(a):\n    return a + 1\n", "def g(x):\n    return x + 2\n"]
    result = compute_uncertainty(outputs)
    assert result["token_jaccard_diversity"] > 0.0
    assert result["ast_node_diversity"] == 0.0


# ---------------------------------------------------------------------------
# Week 4: per-sample self-consistency (SelfCheckGPT-style leave-one-out)
# ---------------------------------------------------------------------------


def test_token_jaccard_self_consistency_ring_hand_computed():
    # Token sets {a,b}, {b,c}, {c,d}. Pairwise Jaccard distances:
    # d01 = 1 - 1/3 = 2/3 ({a,b} vs {b,c})
    # d02 = 1 - 0/4 = 1.0 ({a,b} vs {c,d}, disjoint)
    # d12 = 1 - 1/3 = 2/3 ({b,c} vs {c,d})
    # Leave-one-out means (divide by n-1=2):
    # sample0: (2/3 + 1.0)/2 = 5/6; sample1: (2/3+2/3)/2 = 2/3; sample2: (1.0+2/3)/2 = 5/6
    scores = token_jaccard_self_consistency(["a b", "b c", "c d"])
    assert scores == [pytest.approx(5 / 6), pytest.approx(2 / 3), pytest.approx(5 / 6)]


def test_token_jaccard_self_consistency_two_outputs():
    # n=2: each sample's LOO mean is just the single pairwise distance.
    scores = token_jaccard_self_consistency(["a b", "b c"])
    assert scores == [pytest.approx(2 / 3), pytest.approx(2 / 3)]


def test_token_jaccard_self_consistency_fewer_than_two_outputs():
    assert token_jaccard_self_consistency([]) == []
    assert token_jaccard_self_consistency(["only one"]) == [0.0]


def test_token_cosine_self_consistency_two_outputs():
    # Count vectors {a:1,b:1} vs {b:1,c:1}: dot=1, norms sqrt(2) each -> cos=0.5 -> distance=0.5.
    scores = token_cosine_self_consistency(["a b", "b c"])
    assert scores == [pytest.approx(0.5), pytest.approx(0.5)]


def test_ast_node_self_consistency_structural_pair():
    # Reuses ast_node_diversity's hand-computed pair (normalized Levenshtein 0.6).
    scores = ast_node_self_consistency(["x = 1\n", "def f():\n    return 1\n"])
    assert scores == [pytest.approx(0.6), pytest.approx(0.6)]


def test_ast_node_self_consistency_unparseable_mixed():
    # Same trio as ast_node_diversity's mixed_hand_computed test:
    # d01=0.0 (identical), d02=1.0, d12=1.0 (unparseable).
    # LOO means: sample0 (0+1)/2=0.5; sample1 (0+1)/2=0.5; sample2 (1+1)/2=1.0.
    outputs = ["x = 1", "x = 1", "def f(:"]
    scores = ast_node_self_consistency(outputs)
    assert scores == [pytest.approx(0.5), pytest.approx(0.5), pytest.approx(1.0)]


def test_ast_node_self_consistency_fewer_than_two_outputs():
    assert ast_node_self_consistency([]) == []
    assert ast_node_self_consistency(["x = 1"]) == [0.0]


def test_codebert_cosine_self_consistency_hand_computed():
    # Samples 0 and 2 identical direction (1,0); sample 1 orthogonal (0,1).
    fake = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    with patch.object(uncertainty, "_embed_outputs", return_value=fake):
        scores = codebert_cosine_self_consistency(["a", "b", "c"])
    # distances: d01=1, d02=0, d12=1 -> LOO means: s0=(1+0)/2=0.5; s1=(1+1)/2=1.0; s2=(0+1)/2=0.5
    assert scores == [pytest.approx(0.5), pytest.approx(1.0), pytest.approx(0.5)]


def test_codebert_cosine_self_consistency_fewer_than_two_outputs_skips_model():
    with patch.object(uncertainty, "_embed_outputs") as fake_embed:
        assert codebert_cosine_self_consistency([]) == []
        assert codebert_cosine_self_consistency(["only one"]) == [0.0]
    fake_embed.assert_not_called()


def test_compute_self_consistency_default_keys_and_length():
    outputs = ["a b", "b c", "c d"]
    rows = compute_self_consistency(outputs)
    assert len(rows) == len(outputs)
    for row in rows:
        assert set(row.keys()) == {
            "token_jaccard_self_distance",
            "token_cosine_self_distance",
            "ast_node_self_distance",
        }


def test_compute_self_consistency_include_codebert_adds_key():
    fake = np.array([[1.0, 0.0], [0.0, 1.0]])
    with patch.object(uncertainty, "_embed_outputs", return_value=fake):
        rows = compute_self_consistency(["x = 1", "y = 2"], include_codebert=True)
    assert all("codebert_cosine_self_distance" in row for row in rows)


def test_compute_self_consistency_matches_per_family_functions():
    # The combined function must not silently recompute differently from
    # the standalone per-family functions it wraps.
    outputs = ["a b", "b c", "c d"]
    rows = compute_self_consistency(outputs)
    jaccard = token_jaccard_self_consistency(outputs)
    cosine = token_cosine_self_consistency(outputs)
    ast_scores = ast_node_self_consistency(outputs)
    for i, row in enumerate(rows):
        assert row["token_jaccard_self_distance"] == pytest.approx(jaccard[i])
        assert row["token_cosine_self_distance"] == pytest.approx(cosine[i])
        assert row["ast_node_self_distance"] == pytest.approx(ast_scores[i])


# ---------------------------------------------------------------------------
# Week 4: UncertaintyPredictor
# ---------------------------------------------------------------------------


def _separable_synthetic_data(
    seed: int = 42, n_groups: int = 40, rows_per_group: int = 5
):
    """Linearly separable synthetic (X, y, groups) for predictor mechanics tests."""
    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    return X, y, groups


def test_uncertainty_predictor_rejects_unknown_model_type():
    with pytest.raises(ValueError):
        UncertaintyPredictor(model_type="not_a_real_model")


def test_uncertainty_predictor_feature_importances_before_fit_raises():
    predictor = UncertaintyPredictor(model_type="logistic")
    with pytest.raises(RuntimeError):
        predictor.feature_importances()


@pytest.mark.parametrize("model_type", ["logistic", "random_forest"])
def test_uncertainty_predictor_fit_predict_shapes_and_ranges(model_type):
    X, y, _ = _separable_synthetic_data()
    predictor = UncertaintyPredictor(model_type=model_type, seed=0).fit(
        X, y, feature_names=["a", "b", "c"]
    )
    preds = predictor.predict(X)
    proba = predictor.predict_proba(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1})
    assert proba.shape == y.shape
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    importances = predictor.feature_importances()
    assert set(importances.keys()) == {"a", "b", "c"}


def test_uncertainty_predictor_deterministic_with_fixed_seed():
    X, y, _ = _separable_synthetic_data()
    p1 = UncertaintyPredictor(model_type="random_forest", seed=7).fit(X, y)
    p2 = UncertaintyPredictor(model_type="random_forest", seed=7).fit(X, y)
    assert np.allclose(p1.predict_proba(X), p2.predict_proba(X))


# ---------------------------------------------------------------------------
# Week 4: evaluate_predictor_cv
# ---------------------------------------------------------------------------


def test_evaluate_predictor_cv_recovers_strong_signal():
    # A sanity ceiling: on cleanly separable synthetic data with plenty of
    # groups, leave-problems-out CV should still find near-perfect signal —
    # catches gross wiring regressions without asserting exact numbers.
    X, y, groups = _separable_synthetic_data()
    result = evaluate_predictor_cv(
        X,
        y,
        groups,
        model_type="logistic",
        n_splits=5,
        seed=42,
        feature_names=["a", "b", "c"],
    )
    assert result["summary"]["f1_mean"] > 0.85
    assert result["summary"]["roc_auc_mean"] > 0.9
    assert len(result["fold_metrics"]) == 5
    assert set(result["feature_importances"].keys()) == {"a", "b", "c"}


def test_evaluate_predictor_cv_caps_n_splits_to_n_groups():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(9, 2))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # only 3 distinct groups
    result = evaluate_predictor_cv(X, y, groups, n_splits=10, seed=0)
    assert result["n_splits"] == 3
    assert len(result["fold_metrics"]) == 3


def test_evaluate_predictor_cv_never_splits_a_group_across_train_and_test():
    # Spies on the actual StratifiedGroupKFold.split() calls evaluate_predictor_cv
    # makes, to directly verify the leakage-prevention contract rather than
    # inferring it indirectly from metric values.
    X, y, groups = _separable_synthetic_data(n_groups=12, rows_per_group=4)
    seen_splits = []
    original_split = StratifiedGroupKFold.split

    def spy_split(self, X_, y_, groups_):
        for train_idx, test_idx in original_split(self, X_, y_, groups_):
            seen_splits.append((train_idx, test_idx))
            yield train_idx, test_idx

    with patch.object(StratifiedGroupKFold, "split", spy_split):
        evaluate_predictor_cv(X, y, groups, n_splits=4, seed=0)

    assert len(seen_splits) == 4
    for train_idx, test_idx in seen_splits:
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_evaluate_predictor_cv_handles_single_class_test_fold_gracefully():
    # Force a single-class TEST fold (train keeps both classes, so fitting
    # still succeeds) by monkeypatching the splitter to yield a fixed split
    # directly — StratifiedGroupKFold actively tries to avoid this scenario
    # organically, so this exercises the roc_auc/pr_auc None-fallback branch
    # deterministically instead of fighting the stratifier for one.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 2))
    y = np.array([0, 1] * 10)  # balanced overall
    groups = np.arange(20)  # one row per group: any split is leakage-free

    train_idx = np.arange(4, 20)  # both classes present
    test_idx = np.array([0, 2])  # y[[0, 2]] == [0, 0] -> single class in test

    def fixed_split(self, X_, y_, groups_):
        yield train_idx, test_idx

    with patch.object(StratifiedGroupKFold, "split", fixed_split):
        result = evaluate_predictor_cv(X, y, groups, n_splits=2, seed=0)

    assert result["fold_metrics"][0]["roc_auc"] is None
    assert result["fold_metrics"][0]["pr_auc"] is None
    assert result["summary"]["roc_auc_mean"] is None
    assert result["summary"]["pr_auc_mean"] is None


def test_evaluate_predictor_cv_returns_oof_proba_for_every_row():
    X, y, groups = _separable_synthetic_data()
    result = evaluate_predictor_cv(X, y, groups, n_splits=5, seed=42)
    oof_proba = np.array(result["oof_proba"])
    assert oof_proba.shape == y.shape
    assert np.isfinite(oof_proba).all()  # every row assigned to exactly one test fold
    assert np.all((oof_proba >= 0.0) & (oof_proba <= 1.0))
    assert result["oof_y"] == y.tolist()


# ---------------------------------------------------------------------------
# Week 5: compute_calibration
# ---------------------------------------------------------------------------


def test_compute_calibration_perfectly_calibrated_has_near_zero_ece():
    # Two bins by construction: 10 rows at y_prob=0.2 with 2/10 positive
    # (frac_pos = 0.2, exact match), 10 rows at y_prob=0.8 with 8/10
    # positive (frac_pos = 0.8, exact match) -> both bins' gap is 0.
    y_prob = [0.2] * 10 + [0.8] * 10
    y_true = [1, 1] + [0] * 8 + [1] * 8 + [0, 0]
    result = compute_calibration(y_true, y_prob, n_bins=2, strategy="uniform")
    assert result["ece"] == pytest.approx(0.0, abs=1e-9)
    assert result["mce"] == pytest.approx(0.0, abs=1e-9)
    assert result["n_bins_actual"] == 2


def test_compute_calibration_uniformly_overconfident_has_high_ece():
    # 20 rows, model always says 0.9 confident, but only 2/20 (10%) actually
    # pass -> single bin, gap = |0.10 - 0.90| = 0.80, ECE = MCE = 0.80.
    y_prob = [0.9] * 20
    y_true = [1, 1] + [0] * 18
    result = compute_calibration(y_true, y_prob, n_bins=4, strategy="quantile")
    assert result["n_bins_actual"] == 1  # all predictions identical -> one bin
    assert result["ece"] == pytest.approx(0.8, abs=1e-9)
    assert result["mce"] == pytest.approx(0.8, abs=1e-9)


def test_compute_calibration_brier_score_matches_sklearn():
    from sklearn.metrics import brier_score_loss

    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0.0, 1.0, size=50)
    y_true = (rng.uniform(size=50) < y_prob).astype(int)
    result = compute_calibration(y_true, y_prob)
    assert result["brier_score"] == pytest.approx(
        brier_score_loss(y_true, y_prob), abs=1e-9
    )


def test_compute_calibration_default_bin_count_uses_cube_root_heuristic():
    rng = np.random.default_rng(1)
    y_prob = rng.uniform(0.0, 1.0, size=60)
    y_true = rng.integers(0, 2, size=60)
    result = compute_calibration(y_true, y_prob)
    assert result["n_bins"] == round(60 ** (1 / 3))  # == 4


def test_compute_calibration_drops_nonfinite_oof_rows():
    y_prob = [0.5, np.nan, 0.5, 0.5]
    y_true = [1, 0, 0, 1]
    result = compute_calibration(y_true, y_prob, n_bins=2, strategy="uniform")
    assert result["n_rows"] == 3


def test_compute_calibration_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        compute_calibration([1, 0], [0.5, 0.5, 0.5])


def test_fit_beta_calibrator_fixes_overconfident_probabilities():
    from src.uncertainty import fit_beta_calibrator

    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 2000)
    y = (rng.uniform(size=2000) < p**2).astype(int)  # true prob = p^2: overconfident
    cal = fit_beta_calibrator(y[:1000], p[:1000])
    pre = compute_calibration(y[1000:], p[1000:])["ece"]
    post = compute_calibration(y[1000:], cal(p[1000:]))["ece"]
    assert post < pre / 2  # large, not marginal, improvement


def test_fit_beta_calibrator_near_identity_on_calibrated_input():
    # Kull et al.'s key property vs Platt: the beta family contains the
    # identity, so already-calibrated probabilities are left (near-)unchanged.
    from src.uncertainty import fit_beta_calibrator

    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 2000)
    y = (rng.uniform(size=2000) < p).astype(int)
    cal = fit_beta_calibrator(y[:1000], p[:1000])
    pre = compute_calibration(y[1000:], p[1000:])["ece"]
    post = compute_calibration(y[1000:], cal(p[1000:]))["ece"]
    assert post <= pre + 0.02  # must not uncalibrate


def test_fit_beta_calibrator_single_class_returns_identity():
    from src.uncertainty import fit_beta_calibrator

    cal = fit_beta_calibrator([1, 1, 1], [0.2, 0.5, 0.9])
    assert cal(np.array([0.3, 0.7])) == pytest.approx([0.3, 0.7])


def test_fit_beta_calibrator_handles_extreme_probabilities():
    from src.uncertainty import fit_beta_calibrator

    cal = fit_beta_calibrator([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    out = cal(np.array([0.0, 1.0]))  # must not produce inf/nan via log(0)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_compute_calibration_bin_arrays_are_reliability_diagram_ready():
    y_prob = [0.1, 0.15, 0.85, 0.9]
    y_true = [0, 0, 1, 1]
    result = compute_calibration(y_true, y_prob, n_bins=2, strategy="uniform")
    assert len(result["bin_count"]) == result["n_bins_actual"]
    assert len(result["bin_mean_predicted"]) == result["n_bins_actual"]
    assert len(result["bin_fraction_positive"]) == result["n_bins_actual"]
    assert sum(result["bin_count"]) == 4


# ---------------------------------------------------------------------------
# Week 6: execution-behavior diversity (verdict-free behavioral axis)
# ---------------------------------------------------------------------------


def test_execution_behavior_diversity_identical_signatures_is_zero():
    sigs = [["3", "7"], ["3", "7"], ["3", "7"]]
    assert execution_behavior_diversity(sigs) == 0.0


def test_execution_behavior_diversity_hand_computed_pairs():
    # Pair distances: (a,b): 1/2 differ, (a,c): 2/2, (b,c): 1/2 -> mean = 2/3.
    a = ["1", "2"]
    b = ["1", "9"]
    c = ["8", "9"]
    assert execution_behavior_diversity([a, b, c]) == pytest.approx(2 / 3)


def test_execution_behavior_diversity_identical_failures_graded_default():
    # He et al. Eq. 2 default (a, b, c) = (1, 0.8, 0.6): two samples failing
    # with the SAME exception on every probe are only partially equivalent
    # ("identical errors... need not imply behavioural equivalence") -> c = 0.6.
    sigs = [["error:TypeError", "error:TypeError"]] * 2
    assert execution_behavior_diversity(sigs) == pytest.approx(0.6)


def test_execution_behavior_diversity_legacy_corner_treats_same_error_as_agreement():
    # The (1, 1, 0) corner reproduces the Week 6 overnight behavior:
    # identical error strings = full agreement.
    from src.uncertainty import LEGACY_ERROR_COSTS

    sigs = [["error:TypeError", "error:TypeError"]] * 2
    assert execution_behavior_diversity(sigs, error_costs=LEGACY_ERROR_COSTS) == 0.0


def test_execution_behavior_diversity_graded_error_cases_hand_computed():
    # Probe-wise: (normal '1' vs error) -> a=1.0; (TypeError vs ValueError)
    # -> b=0.8. Pair distance = (1.0 + 0.8) / 2 = 0.9.
    a = ["1", "error:TypeError"]
    b = ["error:IndexError", "error:ValueError"]
    assert execution_behavior_diversity([a, b]) == pytest.approx(0.9)


def test_execution_behavior_diversity_fewer_than_two_samples():
    assert execution_behavior_diversity([]) == 0.0
    assert execution_behavior_diversity([["1"]]) == 0.0


def test_execution_behavior_diversity_empty_probe_set_is_zero():
    assert execution_behavior_diversity([[], [], []]) == 0.0


def test_execution_behavior_self_consistency_outlier_scores_highest():
    # a and b agree everywhere; c disagrees with both on every probe.
    a = ["1", "2"]
    b = ["1", "2"]
    c = ["9", "9"]
    scores = execution_behavior_self_consistency([a, b, c])
    # a: mean(d(a,b)=0, d(a,c)=1) = 0.5; same for b; c: mean(1, 1) = 1.0.
    assert scores == pytest.approx([0.5, 0.5, 1.0])


def test_execution_behavior_self_consistency_fewer_than_two_samples():
    assert execution_behavior_self_consistency([["1"]]) == [0.0]
    assert execution_behavior_self_consistency([]) == []


# ---------------------------------------------------------------------------
# Week 7 — prompt-inclusion confound: metric input representations
# ---------------------------------------------------------------------------

# A HumanEval-shaped prompt: import, blank lines, signature, docstring. The
# docstring carries most of the characters and exactly one AST node, which is
# the asymmetry the whole section exists to handle.
_HE_PROMPT = (
    "from typing import List\n"
    "\n"
    "\n"
    "def f(xs: List[int]) -> int:\n"
    '    """Return the sum.\n'
    "\n"
    "    >>> f([1, 2])\n"
    "    3\n"
    '    """\n'
)
# An MBPP-shaped prompt: module docstring then a bare `def` line, which does
# NOT parse on its own (no body) but does tokenize.
_MBPP_PROMPT = '"""Write a function."""\ndef g(s, ch):\n'


def test_split_prompt_completion_recovers_the_model_output():
    prompt, completion = split_prompt_completion(
        _HE_PROMPT + "    return 1\n", _HE_PROMPT
    )
    assert prompt == _HE_PROMPT
    assert completion == "    return 1\n"


def test_split_prompt_completion_rejects_mismatched_prompt():
    # Failing loudly matters: silently treating the whole string as a
    # completion would corrupt every metric computed from it.
    with pytest.raises(ValueError):
        split_prompt_completion("def other():\n    pass\n", _HE_PROMPT)


def test_strip_prompt_docstring_keeps_imports_and_signature_verbatim():
    stripped = strip_prompt_docstring(_HE_PROMPT)
    assert "from typing import List" in stripped
    assert "def f(xs: List[int]) -> int:" in stripped
    assert "Return the sum" not in stripped


def test_strip_prompt_docstring_handles_unparseable_mbpp_prompt():
    # `def g(s, ch):` with no body is not parseable; tokenize still works,
    # which is exactly why this is token-based rather than ast-based.
    assert strip_prompt_docstring(_MBPP_PROMPT) == "def g(s, ch):\n"


def test_strip_prompt_docstring_keeps_non_docstring_string_literals():
    source = 'x = "keep me"\n'
    assert strip_prompt_docstring(source) == source


def test_strip_prompt_docstring_truncates_trailing_comment_but_keeps_code():
    assert strip_prompt_docstring("x = 1  # drop this\n") == "x = 1\n"


def test_strip_prompt_docstring_returns_untokenizable_input_unchanged():
    broken = "def f(:\n    '\n"
    assert strip_prompt_docstring(broken) == broken


def test_prepare_metric_inputs_representations():
    codes = [_HE_PROMPT + "    return 1\n", _HE_PROMPT + "    return 2\n"]
    assert prepare_metric_inputs(codes, _HE_PROMPT, representation="full") == codes
    assert prepare_metric_inputs(codes, _HE_PROMPT, representation="completion") == [
        "    return 1\n",
        "    return 2\n",
    ]
    signature = prepare_metric_inputs(codes, _HE_PROMPT, representation="signature")
    assert all("Return the sum" not in code for code in signature)
    # The point of "signature": still parseable, unlike bare completions.
    assert parse_failure_rate(signature) == 0.0
    assert (
        parse_failure_rate(
            prepare_metric_inputs(codes, _HE_PROMPT, representation="completion")
        )
        == 1.0
    )


def test_prepare_metric_inputs_rejects_unknown_representation():
    with pytest.raises(ValueError):
        prepare_metric_inputs([_HE_PROMPT], _HE_PROMPT, representation="stripped")


def test_prompt_aware_full_representation_reproduces_compute_uncertainty():
    # Weeks 2-6 reproducibility guard: "full" must be bit-identical to the
    # uncorrected function, or prior numbers stop being comparable.
    codes = [_HE_PROMPT + "    return 1\n", _HE_PROMPT + "    return sum(xs)\n"]
    assert compute_uncertainty_prompt_aware(
        codes, _HE_PROMPT, representation="full"
    ) == compute_uncertainty(codes)


def test_prompt_aware_corrected_raises_token_diversity_above_full():
    # The confound's signature: a shared ~70% prefix deflates token-overlap
    # metrics, so removing it can only increase the measured distance.
    codes = [_HE_PROMPT + "    return 1\n", _HE_PROMPT + "    return sum(xs)\n"]
    full = compute_uncertainty_prompt_aware(codes, _HE_PROMPT, representation="full")
    corrected = compute_uncertainty_prompt_aware(codes, _HE_PROMPT)
    assert corrected["token_jaccard_diversity"] > full["token_jaccard_diversity"]
    assert corrected["token_cosine_diversity"] > full["token_cosine_diversity"]
    # Structural metrics stay defined (the docstring-stripped stub parses).
    assert corrected["parse_failure_rate"] == 0.0


def test_prompt_aware_keeps_the_full_metric_key_set():
    codes = [_HE_PROMPT + "    return 1\n", _HE_PROMPT + "    return 2\n"]
    assert set(compute_uncertainty_prompt_aware(codes, _HE_PROMPT)) == set(
        compute_uncertainty(codes)
    )


def test_prompt_confound_covariates_hand_computed():
    codes = [_HE_PROMPT + "    return 1\n", _HE_PROMPT + "    return 22\n"]
    cov = prompt_confound_covariates(codes, _HE_PROMPT)
    # Completions are "    return 1\n" (13 chars) and "    return 22\n" (14).
    assert cov["mean_completion_chars"] == pytest.approx(13.5)
    assert cov["prompt_chars"] == pytest.approx(len(_HE_PROMPT))
    # Shares are per-sample means of prompt/(prompt+completion), all in (0, 1).
    assert 0.0 < cov["prompt_share_char"] < 1.0
    assert 0.0 < cov["prompt_share_token"] < 1.0
    # The dilution asymmetry this project measured: the prompt is a large
    # share of the characters and a much smaller share of the AST nodes.
    assert cov["prompt_share_ast"] < cov["prompt_share_char"]


def test_prompt_confound_covariates_ast_share_on_unparseable_prompt():
    # MBPP-shaped prompt: unparseable alone, made parseable with a `pass`
    # body whose one node is subtracted back out.
    codes = [_MBPP_PROMPT + "    return s\n", _MBPP_PROMPT + "    return ch\n"]
    cov = prompt_confound_covariates(codes, _MBPP_PROMPT)
    assert not np.isnan(cov["prompt_share_ast"])
    assert 0.0 < cov["prompt_share_ast"] < 1.0


def test_prompt_confound_covariates_empty_input():
    assert prompt_confound_covariates([], _HE_PROMPT) == {}


# ---------------------------------------------------------------------------
# Week 7 — crash-rate control for the execution-behavior family
# ---------------------------------------------------------------------------


def test_execution_crash_rate_hand_computed():
    assert execution_crash_rate(["1", "error:TypeError", "3", "error:timeout"]) == 0.5
    assert execution_crash_rate(["1", "2"]) == 0.0
    assert execution_crash_rate(["error:unrunnable"]) == 1.0


def test_execution_crash_rate_empty_probe_set():
    assert execution_crash_rate([]) == 0.0


def test_nocrash_diversity_ignores_probes_where_either_sample_errored():
    # Probe 0: both ran and agree -> 0. Probe 1: one errored -> dropped.
    # Graded diversity would charge 1.0 for that probe; nocrash sees only
    # probe 0 and reports perfect agreement.
    a = ["1", "2"]
    b = ["1", "error:TypeError"]
    assert execution_behavior_diversity_nocrash([a, b]) == 0.0
    assert execution_behavior_diversity([a, b]) > 0.0


def test_nocrash_diversity_still_sees_real_disagreement():
    a = ["1", "2", "error:TypeError"]
    b = ["9", "2", "error:ValueError"]
    # Surviving probes: 0 (differ -> 1.0) and 1 (agree -> 0.0) => 0.5.
    assert execution_behavior_diversity_nocrash([a, b]) == pytest.approx(0.5)


def test_nocrash_diversity_with_no_co_running_probe_is_zero():
    # Documented conservative convention: no shared evidence is scored as
    # agreement (0.0), not as maximal distance.
    a = ["error:TypeError", "1"]
    b = ["1", "error:ValueError"]
    assert execution_behavior_diversity_nocrash([a, b]) == 0.0


def test_nocrash_diversity_fewer_than_two_samples():
    assert execution_behavior_diversity_nocrash([]) == 0.0
    assert execution_behavior_diversity_nocrash([["1"]]) == 0.0


def test_nocrash_self_consistency_outlier_scores_highest():
    a = ["1", "2"]
    b = ["1", "2"]
    c = ["9", "9"]
    scores = execution_behavior_self_consistency_nocrash([a, b, c])
    assert scores == pytest.approx([0.5, 0.5, 1.0])


def test_nocrash_self_consistency_fewer_than_two_samples():
    assert execution_behavior_self_consistency_nocrash([["1"]]) == [0.0]
    assert execution_behavior_self_consistency_nocrash([]) == []

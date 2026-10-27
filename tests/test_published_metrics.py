"""Tests for the re-implemented published metric definitions (Week 11).

These guard a different property than the rest of the suite: not "does our
metric behave sensibly" but "is this *their* definition". Every assertion
below traces to a formula quoted verbatim in
``reports/week11_published_metric_definitions.md``, and the degenerate
cases exist because two of the four papers specify no behaviour for them —
so our choice has to be pinned somewhere it will break if someone changes
it silently.
"""

import ast
import math

import pytest

from src import published_metrics as pm
from src import uncertainty

ALPHA_A = "def f(x):\n    return x + 1\n"
ALPHA_B = "def f(y):\n    return y + 1\n"
NESTED = "def f(x):\n    if x:\n        return 1\n    return 0\n"
UNPARSEABLE = "def ((("


# --- TSED -------------------------------------------------------------------


def test_tsed_identical_programs_have_zero_diversity():
    assert pm.tsed_diversity([ALPHA_A, ALPHA_A]) == 0.0


def test_tsed_ignores_identifier_renames():
    """Labels are node *types*, so an alpha-rename is structurally identical."""
    assert pm.tsed_diversity([ALPHA_A, ALPHA_B]) == 0.0


def test_tsed_matches_equation_2_by_hand():
    """max{1 - delta / MaxNodes(G1, G2), 0} on a hand-checkable pair.

    Under unit weights, deleting every node of the smaller tree and
    inserting every node of the larger is an upper bound on delta, so
    TSED >= 1 - (n1 + n2)/max(n1, n2), and the ramp keeps it at or above 0.
    The value must also equal 1 - delta/max_nodes exactly for the pair.
    """
    tree_a = uncertainty._ast_tree(ALPHA_A)
    tree_b = uncertainty._ast_tree(NESTED)
    similarity = pm._tsed_pair(tree_a, tree_b, pm.TSED_WEIGHTS_UNIT)
    from zss import distance as zss_distance

    delta = zss_distance(
        tree_a,
        tree_b,
        get_children=lambda t: t[1],
        insert_cost=lambda _n: 1.0,
        remove_cost=lambda _n: 1.0,
        update_cost=lambda a, b: 0.0 if a[0] == b[0] else 1.0,
    )
    max_nodes = max(uncertainty._tree_size(tree_a), uncertainty._tree_size(tree_b))
    assert similarity == pytest.approx(max(1.0 - delta / max_nodes, 0.0))


def test_tsed_ramp_function_clips_at_zero():
    """Eq. 2's ramp: an unparseable member is maximally dissimilar, never < 0."""
    assert (
        pm._tsed_pair(None, uncertainty._ast_tree(ALPHA_A), pm.TSED_WEIGHTS_UNIT) == 0.0
    )
    assert pm.tsed_diversity([ALPHA_A, UNPARSEABLE]) == 1.0


def test_tsed_insert_weight_is_configurable_not_hardcoded():
    """The paper's weights are optimized on Java and it warns Python may differ."""
    paper = pm.tsed_diversity([ALPHA_A, NESTED], weights=pm.TSED_WEIGHTS_PAPER)
    unit = pm.tsed_diversity([ALPHA_A, NESTED], weights=pm.TSED_WEIGHTS_UNIT)
    # Insert is cheaper under the paper's weights, so the distance cannot be larger.
    assert paper <= unit


def test_tsed_single_output_is_zero():
    assert pm.tsed_diversity([ALPHA_A]) == 0.0
    assert pm.tsed_diversity([]) == 0.0


# --- Structural entropy -----------------------------------------------------


def test_subtree_encoding_at_depth_one_is_the_papers_written_form():
    """sigma_struct: sub(v;d) -> (node-type(v), (node-type(c1), ..., node-type(ck)))."""
    root = ast.parse("x = 1")
    encoded = pm._encode_subtree(root, 1, with_lexeme=False)
    assert encoded == ("Module", ("Assign",))


def test_subtree_encoding_value_variant_adds_the_lexeme():
    """sigma_value adds lexeme(v); for ast that is the typed attribute.

    Note the ``Load`` child: ``ast`` materialises ``expr_context`` nodes that a
    tree-sitter parse has no counterpart for. They are kept, deliberately —
    ``uncertainty._ast_tree`` keeps them too, and the pre-registration's H3
    gate compares this module's TSED against that metric to isolate the
    *normalizer*. Dropping ``ctx`` here would make the two differ for an
    unrelated reason and empty the gate of meaning.
    """
    name = ast.parse("foo").body[0].value
    assert pm._encode_subtree(name, 1, with_lexeme=True) == (
        "Name",
        "foo",
        (("Load", ""),),
    )
    assert pm._encode_subtree(name, 1, with_lexeme=False) == ("Name", ("Load",))


def test_depth_changes_the_distribution():
    """d is unspecified by the paper, so it had better be a real parameter."""
    shallow = pm.subtree_distribution(NESTED, depth=1)
    deep = pm.subtree_distribution(NESTED, depth=3)
    assert set(shallow) != set(deep)


def test_jsd_identical_programs_are_zero():
    assert pm.structural_entropy_jsd_diversity([ALPHA_A, ALPHA_A]) == 0.0


def test_sce_identical_programs_are_zero():
    assert pm.structural_entropy_sce_diversity([ALPHA_A, ALPHA_A]) == 0.0


def test_struct_encoding_is_blind_to_renames_and_value_encoding_is_not():
    """The pre-registered H1 mechanism, pinned as a unit test.

    sigma_struct reads node types only, so it inherits the structural
    family's low prompt share; sigma_value reads lexemes, so it inherits
    the token family's high one. If this test ever fails, H1's rationale is
    gone.
    """
    assert (
        pm.structural_entropy_jsd_diversity([ALPHA_A, ALPHA_B], encoding="struct")
        == 0.0
    )
    assert (
        pm.structural_entropy_jsd_diversity([ALPHA_A, ALPHA_B], encoding="value") > 0.0
    )


def test_jsd_is_bounded_by_one_in_base_two():
    """D_JS in [0, 1] with log base 2 — what makes the paper's S_JSD range true."""
    value = pm.structural_entropy_jsd_diversity([ALPHA_A, NESTED])
    assert 0.0 <= value <= 1.0


def test_jsd_hand_computed_on_disjoint_supports():
    """Two distributions with disjoint support have D_JS = 1 bit exactly."""
    p = [1.0, 0.0]
    q = [0.0, 1.0]
    assert pm._jensen_shannon(p, q) == pytest.approx(1.0)


def test_sce_equals_one_for_identical_distributions():
    """SCE(P, Q) = H(Q)/H(P, Q) = 1 when P = Q."""
    vec = [0.5, 0.25, 0.25]
    assert pm._sce(vec, vec) == pytest.approx(1.0)


def test_smoothing_floor_is_applied():
    """Q(u) = max(c_B(u)/n_B, epsilon), applied on the shared support."""
    counts = pm.subtree_distribution(ALPHA_A, depth=1)
    support = sorted(set(counts) | {("Nonexistent", ())}, key=repr)
    smoothed = pm._smoothed(counts, support, 1e-6)
    assert min(smoothed) == pytest.approx(1e-6)


def test_epsilon_is_a_real_parameter():
    """epsilon has no value in the paper, so it must actually move the number."""
    coarse = pm.structural_entropy_jsd_diversity([ALPHA_A, NESTED], epsilon=1e-2)
    fine = pm.structural_entropy_jsd_diversity([ALPHA_A, NESTED], epsilon=1e-9)
    assert coarse != fine


def test_unparseable_sample_is_maximally_divergent():
    """Our declared convention; the paper specifies none."""
    assert pm.structural_entropy_jsd_diversity([ALPHA_A, UNPARSEABLE]) == 1.0
    assert pm.structural_entropy_sce_diversity([ALPHA_A, UNPARSEABLE]) == 1.0


def test_all_unparseable_is_defined_and_maximal():
    assert pm.structural_entropy_jsd_diversity([UNPARSEABLE, UNPARSEABLE]) == 1.0


def test_structural_entropy_single_output_is_zero():
    assert pm.structural_entropy_jsd_diversity([ALPHA_A]) == 0.0
    assert pm.structural_entropy_sce_diversity([]) == 0.0


def test_invalid_encoding_and_depth_raise():
    with pytest.raises(ValueError):
        pm.subtree_distribution(ALPHA_A, encoding="nope")
    with pytest.raises(ValueError):
        pm.subtree_distribution(ALPHA_A, depth=0)


# --- Execution family -------------------------------------------------------


def test_graded_costs_follow_equation_2():
    """a = 1.0 one abnormal, b = 0.8 both abnormal different, c = 0.6 both same."""
    assert pm.graded_semantic_distance(["1"], ["error:ValueError"]) == pytest.approx(
        1.0
    )
    assert pm.graded_semantic_distance(
        ["error:ValueError"], ["error:TypeError"]
    ) == pytest.approx(0.8)
    assert pm.graded_semantic_distance(
        ["error:ValueError"], ["error:ValueError"]
    ) == pytest.approx(0.6)
    assert pm.graded_semantic_distance(["1"], ["1"]) == pytest.approx(0.0)
    assert pm.graded_semantic_distance(["1"], ["2"]) == pytest.approx(1.0)


def test_graded_costs_are_configurable():
    """The paper calls them hyperparameters, so they may not be hard-coded."""
    legacy = pm.graded_semantic_distance(
        ["error:ValueError"], ["error:ValueError"], costs=(1.0, 1.0, 0.0)
    )
    assert legacy == pytest.approx(0.0)


def test_isolation_rule_zeroes_similarity_on_unrunnable():
    """ "if either p_i or p_j contains a syntax error, their similarity is zero"."""
    assert pm.codemue_isolation_distance(["error:unrunnable"], ["1"]) == 1.0
    assert (
        pm.codemue_isolation_distance(
            ["error:unrunnable", "error:unrunnable"], ["1", "1"]
        )
        == 1.0
    )


def test_isolation_rule_both_timeout_contributes_zero():
    """A both-timeout probe adds nothing to W_ij, so it reads as disagreement."""
    assert pm.codemue_isolation_distance(
        ["1", "error:timeout"], ["1", "error:timeout"]
    ) == pytest.approx(0.5)
    # Contrast: the graded rule scores the same pair as agreement-ish.
    assert pm.graded_semantic_distance(
        ["1", "error:timeout"], ["1", "error:timeout"]
    ) == pytest.approx(0.3)


def test_isolation_and_graded_disagree_on_all_crash_pairs():
    """The two published crash conventions pull opposite ways -- finding #3."""
    crash = ["error:ZeroDivisionError"] * 4
    assert pm.codemue_isolation_distance(crash, crash) == pytest.approx(0.0)
    assert pm.graded_semantic_distance(crash, crash) == pytest.approx(0.6)


def test_execution_diversity_and_self_consistency_shapes():
    signatures = [["1", "2"], ["1", "3"], ["error:unrunnable", "error:unrunnable"]]
    assert 0.0 <= pm.graded_semantic_diversity(signatures) <= 1.0
    assert 0.0 <= pm.codemue_isolation_diversity(signatures) <= 1.0
    assert len(pm.graded_semantic_self_consistency(signatures)) == 3
    assert len(pm.codemue_isolation_self_consistency(signatures)) == 3
    # The unrunnable sample must be the most isolated one.
    scores = pm.codemue_isolation_self_consistency(signatures)
    assert scores[2] == max(scores)


def test_execution_metrics_degenerate_on_short_input():
    assert pm.graded_semantic_diversity([["1"]]) == 0.0
    assert pm.codemue_isolation_diversity([]) == 0.0
    assert pm.graded_semantic_self_consistency([["1"]]) == [0.0]
    assert pm.graded_semantic_distance([], []) == 0.0
    assert pm.codemue_isolation_distance([], []) == 1.0


# --- Invariant gate (pre-registration section 9) ----------------------------


def test_graded_wrapper_agrees_with_the_projects_own_implementation():
    """Gate: these differ only by the wrapper, so they must agree exactly."""
    signatures = [
        ["1", "2", "error:ValueError"],
        ["1", "3", "error:ValueError"],
        ["error:TypeError", "2", "5"],
    ]
    assert pm.graded_semantic_diversity(signatures) == pytest.approx(
        uncertainty.execution_behavior_diversity(signatures)
    )

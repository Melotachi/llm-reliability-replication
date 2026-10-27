"""Re-implementations of *published* code-uncertainty metric definitions.

Week 11 (Student A). This module is deliberately separate from
``src/uncertainty.py``: that file is the record of this project's own
metrics, these are other people's. Keeping them apart is part of the
audit's hygiene — nothing here may quietly inherit a convenience this
project adopted, and nothing in ``uncertainty.py`` is modified to make
something here easier.

Every definition below is transcribed from the formula quoted verbatim in
``reports/week11_published_metric_definitions.md``, which was written and
committed *before* this module existed. The parameters each paper leaves
free are exposed as arguments with the paper's own anchor as the default,
never hard-coded, and the grids they are swept over are frozen in
``reports/week11_field_audit_preregistration.md``.

Why this exists (pre-registration §1): every measurement finding in this
project so far came from auditing its *own* pipeline. Re-implementing
published definitions on the same stored samples is what turns a self
audit into a field audit — or fails to, which is also a reportable result.

Two deviations from the sources, declared here rather than buried:

1. **Parser.** TSED specifies tree-sitter and the structural-entropy paper
   names no parser. This module uses the standard library ``ast``, as every
   other structural metric in this project does. The definitions are
   parser-agnostic (both produce typed AST nodes), and using ``ast`` is what
   makes the invariant gates in pre-registration §9 meaningful — a
   ``1 - tsed`` that disagreed with ``tree_edit_distance_diversity`` because
   of the *parser* would tell us nothing about the *normalizer*, which is the
   question. Consequence: node-type vocabularies differ from tree-sitter's,
   so absolute values are not comparable to the papers' own tables. Only
   within-corpus deltas are claimed (pre-registration §10).

2. **Tree edit distance algorithm.** TSED specifies APTED (Pawlik &
   Augsten, 2015, 2016); this uses Zhang-Shasha via ``zss``, already a
   project dependency. Both solve the *same* classical ordered tree edit
   distance problem optimally — APTED is a faster path strategy, not a
   different metric — so ``delta`` is identical and only runtime differs.

Scope note (pre-registration §10): showing that a definition's conclusion
moves under a control **on this corpus** is not showing that the paper's
own reported numbers are wrong on its own corpus. No function here licenses
that claim.
"""

from __future__ import annotations

import ast
import math
from collections import Counter
from itertools import combinations

from zss import distance as zss_distance

from src.uncertainty import _ast_tree, _tree_size

# --- TSED (Song et al., ACL 2024 short, arXiv 2404.08817) -------------------

#: Delete / insert / rename penalty weights. The paper optimizes these on
#: MBXP/**Java** (§5.3: "'Insert' has a sweet spot of 0.8. 'Delete' and
#: 'Rename' operations just keep them in 1.0") and warns explicitly that "it
#: can be different in other programming languages". The Python weights are
#: therefore *unspecified*, which is why this is a swept parameter and not a
#: constant. Order is (delete, insert, rename).
TSED_WEIGHTS_PAPER = (1.0, 0.8, 1.0)
TSED_WEIGHTS_UNIT = (1.0, 1.0, 1.0)

#: Encodings from the structural-entropy paper (arXiv 2508.14288). Both are
#: defined there and both are audited; the paper reports them as variants.
SENT_ENCODINGS = ("struct", "value")

#: The paper gives no value for either the depth bound d or the smoothing
#: constant epsilon anywhere (companion §2.1). These are the pre-registered
#: anchors, not the paper's — the paper has none.
SENT_DEPTH_ANCHOR = 2
SENT_EPSILON_ANCHOR = 1e-6

#: Graded execution-outcome costs, He, Sharma & David (arXiv 2605.09023)
#: Eq. 2, quoted in the companion document: a = 1.0 one outcome abnormal,
#: b = 0.8 both abnormal with different error type, c = 0.6 both abnormal
#: with the same error type.
GRADED_COSTS_PAPER = (1.0, 0.8, 0.6)


def _tsed_pair(
    tree_a: tuple | None,
    tree_b: tuple | None,
    weights: tuple[float, float, float],
) -> float:
    """TSED similarity for one pair of canonical ``_ast_tree`` trees.

    Implements Eq. 1 and Eq. 2 of arXiv 2404.08817:

        delta(G1, G2) = min_{ops} sum_i w(op_i)
        TSED = max{1 - delta / MaxNodes(G1, G2), 0}

    An unparseable member makes the pair maximally dissimilar (TSED 0.0),
    matching the convention every structural metric in this project already
    uses. The paper itself says nothing about unparseable programs — one of
    the four absences recorded in the companion document — so this is our
    declared choice, not theirs.
    """
    if tree_a is None or tree_b is None:
        return 0.0
    delete_w, insert_w, rename_w = weights
    delta = zss_distance(
        tree_a,
        tree_b,
        get_children=lambda tree: tree[1],
        insert_cost=lambda _node: insert_w,
        remove_cost=lambda _node: delete_w,
        update_cost=lambda a, b: 0.0 if a[0] == b[0] else rename_w,
    )
    max_nodes = max(_tree_size(tree_a), _tree_size(tree_b))
    if max_nodes == 0:
        return 1.0
    return max(1.0 - delta / max_nodes, 0.0)


def tsed_diversity(
    outputs: list[str],
    *,
    weights: tuple[float, float, float] = TSED_WEIGHTS_PAPER,
) -> float:
    """Mean pairwise ``1 - TSED`` over N samples — TSED read as a diversity.

    TSED is published as a *similarity* in [0, 1] (1 = identical structures)
    for one prediction against one ground truth. Used across N samples of
    the same problem it becomes a diversity signal, and the aggregation the
    original paper does not specify is taken from the two works that do use
    it that way: "an average of all pairwise distances" (Code Roulette,
    arXiv 2506.10204) and "averaging all pairwise comparisons"
    (arXiv 2508.14288). Returning ``1 - TSED`` puts it on the same
    orientation as every other diversity metric in this project — higher
    means the samples disagree more.

    Args:
        outputs: The N generated code strings for one problem.
        weights: (delete, insert, rename) penalties. Defaults to the
            paper's MBXP/Java optimum; the Python values are unspecified by
            the paper, so callers in the audit sweep this.
    Returns:
        Mean over all N(N-1)/2 unordered pairs; 0.0 for fewer than two
        outputs.
    """
    if len(outputs) < 2:
        return 0.0
    trees = [_ast_tree(code) for code in outputs]
    n = len(outputs)
    n_bad = sum(1 for tree in trees if tree is None)
    # Pairs touching an unparseable sample contribute 1.0 (= 1 - TSED 0.0).
    total = float(n_bad * (n - n_bad) + n_bad * (n_bad - 1) // 2)
    # Zhang-Shasha is super-quadratic and pure Python, so collapse
    # structurally identical samples first and weight by multiplicity —
    # the same cost strategy tree_edit_distance_diversity uses.
    counts: Counter[tuple] = Counter(tree for tree in trees if tree is not None)
    unique = list(counts)
    for i, j in combinations(range(len(unique)), 2):
        similarity = _tsed_pair(unique[i], unique[j], weights)
        total += (1.0 - similarity) * counts[unique[i]] * counts[unique[j]]
    return total / (n * (n - 1) // 2)


# --- Structural entropy (Song et al., ASE 2025 NIER, arXiv 2508.14288) ------


def _lexeme(node: ast.AST) -> str:
    """The ``lexeme(v)`` of the paper's ``sigma_value`` encoding, for ``ast``.

    The paper's encoding is written for a tree-sitter parse, where a leaf's
    lexeme is literally its source text. The standard library ``ast`` keeps
    the same information on typed attributes instead, so this is the
    declared mapping (module docstring, deviation 1). Nodes carrying no
    identifier or literal contribute the empty string, which makes
    ``sigma_value`` collapse to ``sigma_struct`` exactly where there is no
    lexeme to add — the intended behaviour.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.alias):
        return node.name
    if isinstance(node, ast.keyword):
        return node.arg or ""
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return ",".join(node.names)
    return ""


def _encode_subtree(node: ast.AST, depth: int, with_lexeme: bool) -> tuple:
    """Encode ``sub(v; d)`` — the fragment rooted at ``v`` down to depth ``d``.

    Paper's encodings, quoted in the companion document:

        sigma_struct: sub(v;d) -> (node-type(v), (node-type(c1), ..., node-type(ck)))
        sigma_value:  sub(v;d) -> (node-type(v), lexeme(v), (node-type(c1), ...))

    Note what the quoted form actually spells out: only ``v`` and its
    *direct* children, i.e. literally ``d = 1``. The subtree is defined for
    general ``d`` but the encoding is written only for that case, so the
    generalisation used here is the natural recursive one — at ``depth``
    levels each child is itself encoded to ``depth - 1`` — and it
    reproduces the paper's written form exactly at ``depth = 1``. This
    discrepancy was found while implementing, i.e. *after* the
    pre-registration froze; it is recorded in the report that scores the
    audit, never patched into the frozen document.
    """
    label: tuple = (type(node).__name__,)
    if with_lexeme:
        label = (type(node).__name__, _lexeme(node))
    children = list(ast.iter_child_nodes(node))
    if depth <= 1:
        child_labels = tuple(
            (type(child).__name__, _lexeme(child))
            if with_lexeme
            else type(child).__name__
            for child in children
        )
        return label + (child_labels,)
    return label + (
        tuple(_encode_subtree(child, depth - 1, with_lexeme) for child in children),
    )


def subtree_distribution(
    code: str, *, depth: int = SENT_DEPTH_ANCHOR, encoding: str = "struct"
) -> Counter | None:
    """Multiset ``S = {sigma(sub(v;d)) | v in T}`` for one program.

    Args:
        code: One generated program.
        depth: The paper's ``d``. It gives **no value anywhere**, so this is
            a pre-registered anchor, not the paper's default.
        encoding: ``"struct"`` for ``sigma_struct`` (node types only) or
            ``"value"`` for ``sigma_value`` (node types plus lexemes).
    Returns:
        Counter over encoded subtrees, or ``None`` if the program does not
        parse. The paper says nothing about unparseable programs; the
        ``None`` is handled by the callers, which declare their convention.
    """
    if encoding not in SENT_ENCODINGS:
        raise ValueError(f"encoding must be one of {SENT_ENCODINGS}, got {encoding!r}")
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    try:
        root = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None
    with_lexeme = encoding == "value"
    counts: Counter = Counter()
    for node in ast.walk(root):
        counts[_encode_subtree(node, depth, with_lexeme)] += 1
    return counts


def _smoothed(counts: Counter, support: list, epsilon: float) -> list[float]:
    """``Q(u) = max(c_B(u) / n_B, epsilon)`` over a shared support.

    The paper's smoothing rule verbatim; it is applied to both sides so the
    two distributions live on the same support ``U``. Note the rule does not
    renormalize, so the smoothed vectors need not sum to 1 — that is the
    paper's definition and it is kept rather than silently "fixed", since
    quietly renormalizing would make this a re-implementation of a metric
    nobody published.
    """
    total = sum(counts.values())
    if total == 0:
        return [epsilon] * len(support)
    return [max(counts.get(key, 0) / total, epsilon) for key in support]


def _jensen_shannon(p: list[float], q: list[float]) -> float:
    """``D_JS(P||Q) = H(M) - 1/2 [H(P) + H(Q)]``, ``M = 1/2 (P + Q)``.

    Natural logarithm, so the value is in [0, ln 2]; the paper writes ``log``
    without naming a base and asserts ``S_JSD = 1 - D_JS in [0, 1]``, which
    only holds for base 2. Base 2 is therefore used, making the range claim
    in the paper true and the value comparable to their reported figures.
    """

    def entropy(vec: list[float]) -> float:
        return -sum(x * math.log2(x) for x in vec if x > 0.0)

    mixture = [(a + b) / 2.0 for a, b in zip(p, q)]
    return entropy(mixture) - 0.5 * (entropy(p) + entropy(q))


def structural_entropy_jsd_diversity(
    outputs: list[str],
    *,
    depth: int = SENT_DEPTH_ANCHOR,
    encoding: str = "struct",
    epsilon: float = SENT_EPSILON_ANCHOR,
) -> float:
    """Mean pairwise ``D_JS`` over depth-bounded AST subtree distributions.

    arXiv 2508.14288's first metric. The paper reports the *similarity*
    ``S_JSD(P, Q) = 1 - D_JS(P||Q)``; this returns the divergence directly so
    it reads as diversity like every other metric in the project, and
    aggregates by "averaging all pairwise comparisons" as the paper
    specifies.

    Unparseable samples are maximally divergent (1.0) — the project's
    standing convention, declared because the paper has none.

    Args:
        outputs: The N generated programs for one problem.
        depth: ``d``; unspecified by the paper, swept by the audit.
        encoding: ``"struct"`` or ``"value"``; the paper defines both.
        epsilon: smoothing; unspecified by the paper, swept by the audit.
    Returns:
        Mean pairwise divergence in [0, 1]; 0.0 for fewer than two outputs.
    """
    return _pairwise_sent(
        outputs, depth=depth, encoding=encoding, epsilon=epsilon, metric="jsd"
    )


def structural_entropy_sce_diversity(
    outputs: list[str],
    *,
    depth: int = SENT_DEPTH_ANCHOR,
    encoding: str = "struct",
    epsilon: float = SENT_EPSILON_ANCHOR,
) -> float:
    """Mean pairwise ``1 - SCE`` over depth-bounded AST subtree distributions.

    arXiv 2508.14288's second metric, ``SCE(P, Q) = H(Q) / H(P, Q)`` with
    ``H(P, Q) = -sum_u P(u) log Q(u)``, which is a *similarity* in (0, 1]
    (1 when P = Q). Returned as ``1 - SCE`` for the same orientation reason
    as :func:`tsed_diversity`.

    SCE is asymmetric by construction, so each unordered pair is averaged
    over both orderings — the paper says only "averaging all pairwise
    comparisons" and does not address the asymmetry. Declared choice.

    Args and returns: as :func:`structural_entropy_jsd_diversity`.
    """
    return _pairwise_sent(
        outputs, depth=depth, encoding=encoding, epsilon=epsilon, metric="sce"
    )


def _sce(p: list[float], q: list[float]) -> float:
    """``SCE(P, Q) = H(Q) / H(P, Q)`` on a shared, smoothed support."""
    cross = -sum(a * math.log2(b) for a, b in zip(p, q) if b > 0.0)
    self_entropy = -sum(x * math.log2(x) for x in q if x > 0.0)
    if cross <= 0.0:
        # Degenerate: Q puts all mass on one subtree with probability 1, so
        # H(Q) = 0 and H(P, Q) = 0. Identical distributions, hence SCE = 1.
        return 1.0
    return self_entropy / cross


def _pairwise_sent(
    outputs: list[str],
    *,
    depth: int,
    encoding: str,
    epsilon: float,
    metric: str,
) -> float:
    """Shared pairwise driver for the two structural-entropy metrics."""
    if len(outputs) < 2:
        return 0.0
    distributions = [
        subtree_distribution(code, depth=depth, encoding=encoding) for code in outputs
    ]
    n = len(outputs)
    n_bad = sum(1 for dist in distributions if dist is None)
    total = float(n_bad * (n - n_bad) + n_bad * (n_bad - 1) // 2)
    valid = [(i, dist) for i, dist in enumerate(distributions) if dist is not None]
    for (_, dist_a), (_, dist_b) in combinations(valid, 2):
        support = sorted(set(dist_a) | set(dist_b), key=repr)
        p = _smoothed(dist_a, support, epsilon)
        q = _smoothed(dist_b, support, epsilon)
        if metric == "jsd":
            total += _jensen_shannon(p, q)
        else:
            # SCE is asymmetric; average both orderings for an unordered pair.
            total += 1.0 - 0.5 * (_sce(p, q) + _sce(q, p))
    return total / (n * (n - 1) // 2)


# --- Execution-family definitions -------------------------------------------


def _outcome_class(outcome: str) -> tuple[bool, str]:
    """Split a probe outcome into (is_abnormal, error type or value)."""
    if outcome.startswith("error:"):
        return True, outcome[len("error:") :]
    return False, outcome


def graded_semantic_distance(
    signature_a: list[str],
    signature_b: list[str],
    *,
    costs: tuple[float, float, float] = GRADED_COSTS_PAPER,
) -> float:
    """Graded per-probe distance of He, Sharma & David (arXiv 2605.09023) Eq. 2.

    Costs, quoted in the companion document: ``a`` when exactly one of the
    two outcomes is abnormal, ``b`` when both are abnormal with *different*
    error types, ``c`` when both are abnormal with the *same* error type.
    Two normal outcomes cost 0.0 when equal and 1.0 when different.

    This is the same rule the project already applies inside
    ``uncertainty._outcome_distance``; it is re-expressed here so the audit
    reads the published definition through its own module rather than
    through this project's plumbing, and pre-registration §9 gates the two
    against each other.

    Args:
        signature_a, signature_b: Probe outcome lists from
            ``evaluation.behavior_signature``, same length and probe order.
        costs: ``(a, b, c)``; the paper calls these hyperparameters
            "because abnormal executions do not induce a canonical semantic
            distance", so they stay configurable.
    Returns:
        Mean per-probe distance in [0, 1]; 0.0 when there are no probes.
    """
    if not signature_a or not signature_b:
        return 0.0
    one_abnormal, both_diff, both_same = costs
    total = 0.0
    for outcome_a, outcome_b in zip(signature_a, signature_b):
        abnormal_a, value_a = _outcome_class(outcome_a)
        abnormal_b, value_b = _outcome_class(outcome_b)
        if abnormal_a and abnormal_b:
            total += both_same if value_a == value_b else both_diff
        elif abnormal_a or abnormal_b:
            total += one_abnormal
        else:
            total += 0.0 if value_a == value_b else 1.0
    return total / len(signature_a)


def codemue_isolation_distance(
    signature_a: list[str], signature_b: list[str]
) -> float:
    """Code-MUE's Isolation Rule (arXiv 2607.12273 §4.3.2) as a distance.

    Quoted in the companion document: "if either p_i or p_j contains a
    syntax error, their semantic similarity is zero"; and for timeouts, "if
    i != j and both executions exceed the sandbox time limit on input x_k,
    the comparison for that input contributes 0 to W_ij".

    Mapping onto ``evaluation.behavior_signature`` outputs, declared:
    ``error:unrunnable`` on every probe is this corpus's syntax-error
    signal (the sample fails to ``exec`` or has no entry point), and
    ``error:timeout`` is the sandbox limit. Note the direction — this rule
    sends crashed pairs to *maximum* distance, the opposite corner from
    this project's naive ``execution_self_nocrash`` arm, which is exactly
    why it is the right control to run beside it (finding #3).

    Returns:
        ``1 - W_ij`` in [0, 1]; 1.0 when either program is unrunnable
        throughout, and 1.0 when no probe contributes.
    """
    if not signature_a or not signature_b:
        return 1.0
    unrunnable_a = all(outcome == "error:unrunnable" for outcome in signature_a)
    unrunnable_b = all(outcome == "error:unrunnable" for outcome in signature_b)
    if unrunnable_a or unrunnable_b:
        return 1.0
    agreements = 0.0
    for outcome_a, outcome_b in zip(signature_a, signature_b):
        if outcome_a == "error:timeout" and outcome_b == "error:timeout":
            continue  # contributes 0 to W_ij, per the rule
        if outcome_a == outcome_b:
            agreements += 1.0
    return 1.0 - agreements / len(signature_a)


def _mean_pairwise(
    signatures: list[list[str]], pair_fn
) -> float:
    """Mean of ``pair_fn`` over all unordered pairs of behaviour signatures."""
    if len(signatures) < 2:
        return 0.0
    values = [pair_fn(a, b) for a, b in combinations(signatures, 2)]
    return sum(values) / len(values)


def graded_semantic_diversity(
    signatures: list[list[str]],
    *,
    costs: tuple[float, float, float] = GRADED_COSTS_PAPER,
) -> float:
    """Problem-level mean pairwise :func:`graded_semantic_distance`."""
    return _mean_pairwise(
        signatures, lambda a, b: graded_semantic_distance(a, b, costs=costs)
    )


def codemue_isolation_diversity(signatures: list[list[str]]) -> float:
    """Problem-level mean pairwise :func:`codemue_isolation_distance`."""
    return _mean_pairwise(signatures, codemue_isolation_distance)


def _self_distances(signatures: list[list[str]], pair_fn) -> list[float]:
    """Per-sample leave-one-out mean distance to the other N-1 samples."""
    n = len(signatures)
    if n < 2:
        return [0.0] * n
    return [
        sum(pair_fn(signatures[i], signatures[j]) for j in range(n) if j != i)
        / (n - 1)
        for i in range(n)
    ]


def graded_semantic_self_consistency(
    signatures: list[list[str]],
    *,
    costs: tuple[float, float, float] = GRADED_COSTS_PAPER,
) -> list[float]:
    """Per-sample counterpart of :func:`graded_semantic_diversity`."""
    return _self_distances(
        signatures, lambda a, b: graded_semantic_distance(a, b, costs=costs)
    )


def codemue_isolation_self_consistency(signatures: list[list[str]]) -> list[float]:
    """Per-sample counterpart of :func:`codemue_isolation_diversity`."""
    return _self_distances(signatures, codemue_isolation_distance)

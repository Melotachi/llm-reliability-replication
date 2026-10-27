"""Uncertainty estimation metrics over multi-sample code generations.

Student A owns this file (see CLAUDE.md Section 6.2).

Week 2 scope (CLAUDE.md Section 10.1 metric progression):
  - lexical_diversity: improved from Week 1's exact-string uniqueness to
    whitespace-insensitive lexeme-sequence uniqueness. Motivated by the
    Week 1 finding that exact-string uniqueness read 1.00 on every problem
    (at temperature 0.7 a model essentially never repeats a byte-identical
    completion), so the metric carried zero information.
  - token_jaccard_diversity: new — Known Issue #9's upgrade path.
  - ast_node_diversity: new — Known Issue #10's structural replacement for
    pairwise_distance.
  - pairwise_distance: retained *unchanged* as the naive Week 1 baseline so
    correlation plots can show whether the structural metric actually beats
    it. Do not use it in new code.

Week 3 scope (CLAUDE.md Section 10.1 metric progression):
  - tree_edit_distance_diversity: true tree edit distance (Zhang-Shasha,
    via the zss library) over ast.parse() trees. Closes the documented
    Week 2 blind spot of ast_node_diversity: pre-order flattening discards
    nesting, so nesting-only differences scored 0.0 there.
  - codebert_cosine_diversity: semantic channel — mean pairwise cosine
    distance between mean-pooled CodeBERT embeddings (Feng et al.). The
    only metric that reads meaning rather than surface/syntactic form, and
    the only one that still yields signal on unparseable samples. Opt-in
    through compute_uncertainty(include_codebert=True) because it loads a
    ~500 MB HuggingFace encoder on first use.

Literature-grounded extensions (added post-Week-3, same weeks' scope —
CLAUDE.md Section 10.1 explicitly asks every metric to trace to a paper):
  - token_cosine_diversity (Week 2 family): cosine distance over token
    *count* vectors — the classical Vector Space Model (Salton & McGill,
    1983), and the same "average a pairwise lexical-similarity function
    over samples" design as the Lexical Similarity baseline in Lin et al.
    (2024, TMLR, "Generating with Confidence: Uncertainty Quantification
    for Black-box Large Language Models"). Closes token_jaccard_diversity's
    multiplicity blind spot: Jaccard operates on token *sets*, so samples
    that differ only in how often a token repeats score 0.0.
  - codebert_bertscore_diversity (Week 3 family): greedy per-token
    embedding alignment (BERTScore, Zhang et al., 2020, ICLR; code-adapted
    as CodeBERTScore, Zhou et al., 2023, EMNLP) instead of one mean-pooled
    vector per sample. SelfCheckGPT (Manakul et al., 2023, EMNLP) uses this
    exact token-alignment similarity, averaged across resampled
    generations, as a black-box hallucination/uncertainty signal — this
    file already does the "compare each sample to the others" part for
    every metric; this is that idea applied to per-token semantics instead
    of sentence-level pooling.

All diversity metrics share one reading: higher value = the N samples
disagree more = the model is less self-consistent = (hypothesis) the
samples are more likely wrong. This is the self-consistency insight of
Wang et al. transferred to code generation (CLAUDE.md Section 10.1).

Week 4 scope (CLAUDE.md Section 10.1: "Build uncertainty predictor (ML model
for pass/fail prediction)"):
  - Self-consistency section (compute_self_consistency and its per-family
    helpers): every metric above is a single *problem-level* number shared
    by all N samples of that problem — useful for ranking problems, but it
    cannot tell samples of the *same* problem apart, since they'd all get
    the identical feature value despite different pass/fail labels. This
    section adds *per-sample* features instead: sample i's leave-one-out
    mean distance to the other N-1 samples of its problem, for each of the
    four families the Week 3 addendum's redundancy heatmap identified as
    genuinely non-overlapping (lexical/Jaccard, lexical/cosine, structural/
    AST, semantic/CodeBERT). Directly the SelfCheckGPT (Manakul et al.,
    2023, EMNLP) mechanism: score each individual sampled output by its
    consistency against the *other* sampled outputs, rather than
    aggregating all samples into one number.
  - UncertaintyPredictor + evaluate_predictor_cv: sklearn binary classifiers
    (logistic regression, random forest) over either feature family, with
    leave-problems-out (StratifiedGroupKFold) cross-validation so no
    problem's rows appear in both train and test — the Week 2/3 reports'
    "point-biserial optimism" caveat about correlated per-problem rows,
    upgraded here from a caveat into an enforced evaluation constraint.
    evaluate_predictor_cv also returns "oof_proba"/"oof_y": every row's
    out-of-fold predicted probability, for Week 5's calibration analysis.

Week 5 scope (CLAUDE.md Section 10.1: "Calibration: Expected Calibration
Error (ECE), reliability diagram"):
  - compute_calibration: is UncertaintyPredictor.predict_proba() actually
    trustworthy as a probability, not just a good ranking signal? Weeks 2-4
    only ever asked whether an uncertainty score correlates with pass/fail;
    this asks whether "70% confident" means "passes about 70% of the time."
    Bin count and strategy are chosen for this project's small calibration
    sets specifically (round(n^(1/3)) bins, quantile/equal-mass binning —
    see the function docstring for the citations), not the field's
    vision-benchmark-scale defaults.
"""

from __future__ import annotations

import ast
import io
import math
import re
import tokenize
from collections import Counter
from itertools import combinations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from zss import simple_distance

from src.utils import get_logger

logger = get_logger(__name__)

# Section 6.2 token definition for token_jaccard_diversity: "split on
# whitespace and punctuation" — i.e. both act as separators, so a token is
# a maximal run of word characters (identifiers, keywords, number literals).
_WORD_TOKEN_RE = re.compile(r"\w+")

# Lexeme definition for the improved lexical_diversity: whitespace is a
# separator, but each punctuation character is *kept* as its own lexeme, so
# `a+b` and `a-b` stay distinct while `a + b` and `a+b` collapse.
_LEXEME_RE = re.compile(r"\w+|[^\w\s]")

# Default embedding model for codebert_cosine_diversity (Feng et al.,
# CodeBERT). Configurable per call — never hardcoded at call sites
# (CLAUDE.md Constraint #2); the pipeline exposes --codebert-model.
DEFAULT_CODEBERT_MODEL = "microsoft/codebert-base"

# RoBERTa positional-embedding limit; longer candidates are truncated.
_MAX_CODEBERT_TOKENS = 512

# Lazy per-model-name cache so repeated metric calls (one per problem) load
# the ~500 MB encoder exactly once per process.
_CODEBERT_CACHE: dict[str, tuple] = {}

# Mean-pooled embeddings, cached per (model_name, outputs): codebert_cosine_diversity
# and codebert_cosine_self_consistency both embed the same per-problem samples
# through _embed_outputs, so without this every problem pays for the encoder
# forward pass twice for numerically identical input. Unbounded is fine at this
# project's per-run scale (a full run's samples are a few hundred to a few
# thousand embedding vectors, each hidden_size floats — single-digit MB); the
# size cap below is a defensive backstop, not an expected code path.
_EMBEDDING_CACHE: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
_EMBEDDING_CACHE_MAX_ENTRIES = 20_000


def _word_token_set(code: str) -> set[str]:
    """Section 6.2 tokens as a set: word-character runs, punctuation dropped."""
    return set(_WORD_TOKEN_RE.findall(code))


def _word_token_counts(code: str) -> Counter[str]:
    """Section 6.2 tokens as a multiset: same tokens as _word_token_set,
    but repetitions are preserved instead of collapsed."""
    return Counter(_WORD_TOKEN_RE.findall(code))


def _lexeme_sequence(code: str) -> tuple[str, ...]:
    """Ordered lexemes (words + individual punctuation), whitespace-insensitive."""
    return tuple(_LEXEME_RE.findall(code))


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    """Jaccard distance 1 - |A∩B|/|A∪B|; two empty sets count as identical."""
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def _cosine_distance_from_counts(a: Counter[str], b: Counter[str]) -> float:
    """Cosine distance 1 - cos(a, b) between two token-count vectors.

    Same edge-case convention as _jaccard_distance: two empty counters
    count as identical (0.0); one empty and one non-empty is maximally
    distant (1.0), since the dot product is 0 whenever either vector is
    all-zero. Counts are non-negative, so unlike embedding cosine
    distance (codebert_cosine_diversity), this is mathematically always
    in [0.0, 1.0]: a non-negative dot product can never exceed the
    product of the norms it is divided by, and can never go below 0.
    Clamped defensively (min/max) so floating-point rounding on the
    sqrt/divide (e.g. identical vectors landing at 1e-16 instead of
    exactly 0) can never push the result outside that range.
    """
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    dot = sum(count * b.get(token, 0) for token, count in a.items())
    norm_a = math.sqrt(sum(count * count for count in a.values()))
    norm_b = math.sqrt(sum(count * count for count in b.values()))
    return min(1.0, max(0.0, 1.0 - dot / (norm_a * norm_b)))


def _ast_node_sequence(code: str) -> tuple[str, ...] | None:
    """Pre-order sequence of AST node-type names, or None if unparseable."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None
    # Iterative pre-order DFS so pathologically nested generations cannot
    # blow the interpreter recursion limit the way a recursive walk could.
    sequence: list[str] = []
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        sequence.append(type(node).__name__)
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return tuple(sequence)


def _normalized_levenshtein(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Levenshtein distance over two symbol sequences, divided by max length.

    max(len(a), len(b)) is an upper bound on the edit distance (substitute
    the overlap, insert/delete the rest), so the result is always in
    [0.0, 1.0]. Two empty sequences are identical (0.0).
    """
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    # Standard two-row dynamic program, O(len(a) * len(b)).
    previous = list(range(len(b) + 1))
    for i, sym_a in enumerate(a, start=1):
        current = [i]
        for j, sym_b in enumerate(b, start=1):
            cost = 0 if sym_a == sym_b else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1] / max(len(a), len(b))


def _ast_tree(code: str) -> tuple | None:
    """Canonical AST as nested (node_type, children_tuple) pairs; None if unparseable.

    Hashable by construction, so structurally identical samples (e.g.
    alpha-renames — labels are node *types* only) deduplicate to the same
    value. Built with an explicit stack (post-order) for the same
    recursion-safety reason as _ast_node_sequence.
    """
    try:
        root = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None
    built: dict[int, tuple] = {}
    stack: list[tuple[ast.AST, bool]] = [(root, False)]
    while stack:
        node, children_done = stack.pop()
        if children_done:
            children = tuple(built[id(child)] for child in ast.iter_child_nodes(node))
            built[id(node)] = (type(node).__name__, children)
        else:
            stack.append((node, True))
            stack.extend((child, False) for child in ast.iter_child_nodes(node))
    return built[id(root)]


def _tree_size(tree: tuple) -> int:
    """Node count of an _ast_tree() canonical tree (iterative)."""
    size = 0
    stack = [tree]
    while stack:
        _, children = stack.pop()
        size += 1
        stack.extend(children)
    return size


def _load_codebert(model_name: str) -> tuple:
    """Load-and-cache (tokenizer, encoder) on first use; encoder in eval mode."""
    if model_name not in _CODEBERT_CACHE:
        from transformers import AutoModel, AutoTokenizer  # deferred: slow import

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model %s on %s (one-time)", model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder = AutoModel.from_pretrained(model_name)
        encoder.to(device)
        encoder.eval()
        _CODEBERT_CACHE[model_name] = (tokenizer, encoder)
    return _CODEBERT_CACHE[model_name]


def _embed_outputs(outputs: list[str], model_name: str) -> np.ndarray:
    """Embed outputs as mean-pooled last hidden states, one row per sample.

    Mean pooling over non-padding tokens rather than [CLS]: without
    similarity fine-tuning, the [CLS] vector is a known-poor sequence
    representation, and mean pooling is the standard alternative.

    Cached per (model_name, outputs) in _EMBEDDING_CACHE (see its module-
    level comment): a pure, deterministic forward pass, so a cache hit is
    exactly the array a fresh call would return.
    """
    cache_key = (model_name, tuple(outputs))
    cached = _EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if len(_EMBEDDING_CACHE) >= _EMBEDDING_CACHE_MAX_ENTRIES:
        _EMBEDDING_CACHE.clear()  # defensive backstop; see module-level comment
    tokenizer, encoder = _load_codebert(model_name)
    device = next(encoder.parameters()).device
    encoded = tokenizer(
        outputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_MAX_CODEBERT_TOKENS,
    ).to(device)
    with torch.no_grad():
        hidden = encoder(**encoded).last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    result = pooled.cpu().numpy()
    _EMBEDDING_CACHE[cache_key] = result
    return result


def _embed_tokens(outputs: list[str], model_name: str) -> list[np.ndarray]:
    """Per-sample, per-token L2-normalized embeddings (special/pad tokens dropped).

    Unlike _embed_outputs' single mean-pooled vector per sample, this
    keeps one row per real (non-special, non-padding) token, letting
    codebert_bertscore_diversity align tokens instead of collapsing each
    sample to one point. Real tokens are attention_mask==1 positions that
    are not the CLS/SEP ids specifically (rather than relying on
    tokenizer.get_special_tokens_mask, whose padding behavior is not
    guaranteed by the HuggingFace API across tokenizers). Rows are
    normalized here, not at call sites, so downstream dot products are
    directly cosine similarities.

    Args:
        outputs: List of N generated code strings.
        model_name: HuggingFace encoder identifier (see _load_codebert).
    Returns:
        List of N arrays, each shape (n_tokens_i, hidden_dim); n_tokens_i
        can be 0 for a sample with no real tokens (e.g. an empty string).
    """
    tokenizer, encoder = _load_codebert(model_name)
    device = next(encoder.parameters()).device
    encoded = tokenizer(
        outputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_MAX_CODEBERT_TOKENS,
    ).to(device)
    with torch.no_grad():
        hidden = encoder(**encoded).last_hidden_state
    input_ids = encoded["input_ids"]
    real_token = encoded["attention_mask"] == 1
    for special_id in (tokenizer.cls_token_id, tokenizer.sep_token_id):
        if special_id is not None:
            real_token &= input_ids != special_id
    per_sample = []
    for i in range(hidden.shape[0]):
        vectors = hidden[i][real_token[i]]
        norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-12)
        per_sample.append((vectors / norms).cpu().numpy())
    return per_sample


def _bertscore_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - BERTScore F1 (Zhang et al., 2020) between two token-embedding sets.

    a, b: (n_tokens, dim) arrays of L2-normalized token embeddings (see
    _embed_tokens), treating a as "candidate" and b as "reference".
    Precision is the mean, over a's tokens, of each token's best (max
    cosine similarity) match in b; recall is the same with a and b
    swapped; F1 is their harmonic mean — BERTScore's greedy token-
    alignment formulation, without the reference-corpus IDF weighting (no
    reference corpus exists here — every sample is compared against every
    other; this is the same unweighted simplification CodeBERTScore
    offers as an option). F1 is symmetric in (a, b) by construction
    (swapping a and b swaps precision and recall, and harmonic mean is
    symmetric in its two arguments), so this induces a valid pairwise
    distance.

    A sample with zero real tokens (e.g. an empty string) has no
    embeddings to match against, so that pair is defined as maximally
    distant (1.0) — same "no structure to compare" convention as
    ast_node_diversity's unparseable case.
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 1.0
    similarity = a @ b.T  # (n_a, n_b) cosine similarities (both unit rows)
    precision = float(similarity.max(axis=1).mean())
    recall = float(similarity.max(axis=0).mean())
    if precision + recall <= 0:
        return 1.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return 1.0 - f1


def lexical_diversity(outputs: list[str]) -> float:
    """Fraction of unique outputs among N samples.

    Range: [1/N, 1.0]. Higher = more diverse = higher uncertainty.
    Baseline metric; intentionally simple.

    Week 2 improvement: uniqueness is computed over whitespace-insensitive
    lexeme sequences (see _LEXEME_RE) instead of raw strings, so two samples
    that differ only in spacing, blank lines, or trailing whitespace no
    longer count as distinct. Operators and other punctuation still count
    (`a+b` != `a-b`). Week 1's exact-string version saturated at 1.00 on
    all 10 problems, carrying zero signal.

    Known limitation (accepted, documented): lexemes ignore *all*
    whitespace, including indentation, so two samples that differ only in
    block structure collapse together. Catching that is exactly
    ast_node_diversity's job, not this baseline's.

    Args:
        outputs: List of N generated code strings for one problem.
    Returns:
        unique_count / N; 0.0 for an empty list.
    """
    if not outputs:
        return 0.0
    unique = len({_lexeme_sequence(code) for code in outputs})
    return unique / len(outputs)


def token_jaccard_diversity(outputs: list[str]) -> float:
    """Average pairwise Jaccard distance over token sets.

    Tokens: split on whitespace and punctuation (Section 6.2 contract), so
    a token is a run of word characters — identifiers, keywords, numeric
    literals — and operators/brackets are discarded. More sensitive than
    exact-string lexical_diversity: samples sharing most of their
    vocabulary score close to 0 even when no two are identical.
    Range: [0.0, 1.0]. Higher = more diverse = higher uncertainty.

    Edge cases (documented design decisions): fewer than two outputs give
    0.0 (a single sample cannot disagree with itself); a pair of outputs
    with no tokens at all counts as identical (distance 0.0).

    Args:
        outputs: List of N generated code strings for one problem.
    Returns:
        Mean Jaccard distance over all N*(N-1)/2 unordered pairs.
    """
    if len(outputs) < 2:
        return 0.0
    token_sets = [_word_token_set(code) for code in outputs]
    distances = [
        _jaccard_distance(token_sets[i], token_sets[j])
        for i, j in combinations(range(len(token_sets)), 2)
    ]
    return sum(distances) / len(distances)


def token_cosine_diversity(outputs: list[str]) -> float:
    """Average pairwise cosine distance over token *count* vectors.

    Same Section 6.2 token definition as token_jaccard_diversity (word-
    character runs; punctuation is a separator), but the classical Vector
    Space Model (Salton & McGill, 1983) instead of set membership: each
    sample becomes a token -> count vector, and pair distance is
    1 - cos(theta) between the two count vectors. Cosine similarity is
    scale-invariant, so raw counts (not term frequencies) are used
    directly — normalizing per-sample would not change the result.

    Motivation (why this is not redundant with token_jaccard_diversity):
    Jaccard operates on token *sets*, so repetition is invisible — "a a a
    b" and "a b b b" have the identical token set {a, b} and score 0.0
    Jaccard distance even though one repeats `a` three times and the
    other repeats `b` three times (see the regression test). Cosine
    distance over count vectors sees this. This mirrors, in the lexical
    family, the same shape of blind spot that motivated
    tree_edit_distance_diversity in the structural family: a coarser
    representation (set / flattened sequence) can hide real disagreement
    that a finer one (multiset / tree) recovers.

    Literature grounding: averaging a pairwise lexical-similarity function
    over multiple sampled generations as an uncertainty/self-consistency
    signal is the "Lexical Similarity" baseline of Lin et al. (2024, TMLR,
    "Generating with Confidence: Uncertainty Quantification for Black-box
    Large Language Models"); cosine similarity over count vectors is the
    classical Vector Space Model (Salton & McGill, 1983). This function is
    that family's second member alongside token_jaccard_diversity, so the
    Week 2 correlation analysis compares a presence/absence lexical metric
    against a frequency-weighted one head-to-head.

    Range: [0.0, 1.0] — unlike codebert_cosine_diversity's embedding
    cosine distance, token counts are non-negative, so the pairwise cosine
    similarity here can never be negative (see _cosine_distance_from_counts).
    Higher = more diverse = higher uncertainty.

    Args:
        outputs: List of N generated code strings for one problem.
    Returns:
        Mean cosine distance over all N*(N-1)/2 unordered pairs; 0.0 if
        fewer than two outputs.
    """
    if len(outputs) < 2:
        return 0.0
    counts = [_word_token_counts(code) for code in outputs]
    distances = [
        _cosine_distance_from_counts(counts[i], counts[j])
        for i, j in combinations(range(len(counts)), 2)
    ]
    return sum(distances) / len(distances)


def ast_node_diversity(outputs: list[str]) -> float:
    """Average pairwise distance between AST node-type sequences.

    Each sample is parsed with ast.parse() and flattened to its pre-order
    sequence of node-type names (e.g. Module, FunctionDef, Return, ...).
    Node types deliberately ignore identifiers and constant values, so two
    samples that differ only in variable names or literals have distance
    0.0 — this metric isolates *structural* disagreement, complementing the
    lexical metrics. Pair distance is Levenshtein edit distance over the
    two sequences normalized by the longer length, which bounds every pair
    in [0.0, 1.0]; this is the standard normalized edit distance and the
    Week 2 stepping stone toward Week 3's true tree edit distance.

    Known limitation (accepted, documented, tested): flattening the tree
    to a pre-order type sequence discards nesting, so two programs whose
    node types match in order but nest differently (e.g. a trailing call
    moved out of an if-body) score 0.0. Week 3's tree edit distance closes
    exactly this gap.

    Unparseable samples: per the Section 6.2 contract, a sample that fails
    to parse is treated as maximally distant (1.0) from all valid samples.
    Design decision (contract is silent here): a pair of *two* unparseable
    samples also scores 1.0 — a parse failure yields no structure to
    compare, and rewarding two broken samples for both being broken would
    lower uncertainty exactly where confidence is least justified.

    Range: [0.0, 1.0] (normalized by max possible distance).
    Higher = more diverse = higher uncertainty.

    Args:
        outputs: List of N generated code strings for one problem.
    Returns:
        Mean normalized distance over all unordered pairs; 0.0 if fewer
        than two outputs.
    """
    if len(outputs) < 2:
        return 0.0
    sequences = [_ast_node_sequence(code) for code in outputs]
    distances = []
    for i, j in combinations(range(len(sequences)), 2):
        if sequences[i] is None or sequences[j] is None:
            distances.append(1.0)
        else:
            distances.append(_normalized_levenshtein(sequences[i], sequences[j]))
    return sum(distances) / len(distances)


def tree_edit_distance_diversity(outputs: list[str]) -> float:
    """Average pairwise normalized tree edit distance between sample ASTs.

    Week 3 upgrade of ast_node_diversity (CLAUDE.md Section 10.1): each
    pair of ast.parse() trees is compared with true Zhang-Shasha tree edit
    distance (zss library) under unit costs for insert/delete/relabel,
    labels being node-*type* names. Nesting differences that pre-order
    flattening hides now count: the documented Week 2 blind spot — moving
    a trailing call out of an if-body scored 0.0 under ast_node_diversity
    — scores > 0 here. Like ast_node_diversity, node types ignore
    identifiers and constant values, so alpha-renamed code still has
    distance 0.0.

    Normalization (documented design decision): each pair is divided by
    n1 + n2, its two trees' node counts. Deleting all of one tree and
    inserting all of the other always costs exactly n1 + n2, so this is a
    guaranteed upper bound and every pair lands in [0.0, 1.0]. Unlike
    sequence Levenshtein, max(n1, n2) is NOT a valid bound for trees (an
    n-node star vs an n-node chain needs ~2n operations, since siblings
    cannot map onto an ancestor line), hence the sum. Because both roots
    always carry the same label (Module), a parseable pair is in fact
    strictly below 1.0 — the value 1.0 is reserved for pairs involving an
    unparseable sample (same contract semantics and two-unparseable design
    decision as ast_node_diversity: parse failure is strictly worse than
    any structural disagreement).

    Cost note: Zhang-Shasha is super-quadratic per pair and pure Python,
    making this the most expensive syntactic metric, so samples are
    deduplicated by canonical tree first (alpha-renames and whitespace
    variants collapse) and each unique pair is computed once, weighted by
    multiplicity.

    Range: [0.0, 1.0]. Higher = more diverse = higher uncertainty.

    Args:
        outputs: List of N generated code strings for one problem.
    Returns:
        Mean normalized TED over all N*(N-1)/2 unordered pairs; 0.0 if
        fewer than two outputs.
    """
    if len(outputs) < 2:
        return 0.0
    trees = [_ast_tree(code) for code in outputs]
    n = len(outputs)
    n_bad = sum(1 for tree in trees if tree is None)
    # Every pair touching an unparseable sample contributes 1.0.
    total = float(n_bad * (n - n_bad) + n_bad * (n_bad - 1) // 2)
    counts: dict[tuple, int] = {}
    for tree in trees:
        if tree is not None:
            counts[tree] = counts.get(tree, 0) + 1
    unique = list(counts)
    sizes = [_tree_size(tree) for tree in unique]
    for i, j in combinations(range(len(unique)), 2):
        edit_distance = simple_distance(
            unique[i],
            unique[j],
            get_children=lambda tree: tree[1],
            get_label=lambda tree: tree[0],
        )
        normalized = edit_distance / (sizes[i] + sizes[j])
        total += normalized * counts[unique[i]] * counts[unique[j]]
    return total / (n * (n - 1) // 2)


def codebert_cosine_diversity(
    outputs: list[str], model_name: str = DEFAULT_CODEBERT_MODEL
) -> float:
    """Average pairwise cosine distance between CodeBERT sample embeddings.

    The Week 3 semantic channel (CLAUDE.md Section 10.1; Feng et al.,
    CodeBERT): every other metric measures surface or syntactic form, so
    two similar-looking implementations of *different* algorithms read as
    agreement there. Embeddings from a code-pretrained encoder capture
    what the code says rather than how it is spelled; and because the
    encoder consumes raw text, this is also the only diversity metric that
    still yields signal on unparseable samples instead of a pinned 1.0.

    Embedding: mean-pooled last hidden state over non-padding tokens (see
    _embed_outputs), inputs truncated at the encoder's 512-token limit.
    Pair distance is cosine distance 1 - cos(a, b).

    Range: theoretically [0.0, 2.0]; in practice transformer embeddings
    are anisotropic (pairwise cosine similarity is high even between
    unrelated code), so observed values sit far below 1.0 and should be
    read comparatively across problems, not as absolute dissimilarity.

    Determinism: a pure forward pass with no sampling, so no seed
    parameter is needed (CLAUDE.md Constraint #4 covers randomness only).

    Args:
        outputs: List of N generated code strings for one problem.
        model_name: HuggingFace encoder identifier. Default
            DEFAULT_CODEBERT_MODEL ("microsoft/codebert-base"); ~500 MB
            download on first use, cached in-process afterwards.
    Returns:
        Mean cosine distance over all N*(N-1)/2 unordered pairs; 0.0 if
        fewer than two outputs.
    """
    if len(outputs) < 2:
        return 0.0
    embeddings = _embed_outputs(outputs, model_name)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.clip(norms, 1e-12, None)
    similarity = unit @ unit.T
    distances = [
        1.0 - float(similarity[i, j]) for i, j in combinations(range(len(outputs)), 2)
    ]
    return float(np.mean(distances))


def codebert_bertscore_diversity(
    outputs: list[str], model_name: str = DEFAULT_CODEBERT_MODEL
) -> float:
    """Average pairwise BERTScore distance between per-token CodeBERT embeddings.

    codebert_cosine_diversity collapses each sample to one mean-pooled
    vector before comparing; this metric instead keeps every token's
    embedding and aligns them greedily, token by token (BERTScore, Zhang
    et al., 2020, ICLR, "BERTScore: Evaluating Text Generation with BERT";
    code-adapted as CodeBERTScore, Zhou et al., 2023, EMNLP). SelfCheckGPT
    (Manakul et al., 2023, EMNLP) uses exactly this token-alignment
    similarity, averaged over resampled generations, as a black-box
    hallucination/uncertainty signal — the same "compare each sample to
    the others" design this file already uses for every metric, applied
    to the token-level semantic channel instead of the sentence-level one.

    Motivation: mean pooling can wash out a localized disagreement (e.g.
    two samples that agree on 90% of tokens but diverge on one critical
    call) into a small shift in one aggregate vector; per-token
    max-similarity matching preserves it. This also gives a second,
    independent read on the Week 3 report's Observation 4 concern that the
    shared prompt prefix inflates codebert_cosine_diversity — greedy
    per-token matching does not average the divergent tokens away with
    the (large, shared) agreeing ones the way mean pooling does.

    Pair distance is 1 - F1, where precision/recall/F1 follow BERTScore's
    definition (see _bertscore_distance); no IDF weighting, since there is
    no reference corpus here. Determinism: forward pass only, no sampling
    — no seed parameter, same as codebert_cosine_diversity.

    Range: theoretically up to 2.0 like codebert_cosine_diversity (cosine
    similarity is in [-1, 1], so F1 built from per-token max-similarities
    is too); in practice higher than the mean-pooled metric's observed
    range, since greedy max-matching does not smooth divergent tokens away
    by averaging.

    Args:
        outputs: List of N generated code strings for one problem.
        model_name: HuggingFace encoder identifier. Default
            DEFAULT_CODEBERT_MODEL; shares the _load_codebert cache with
            codebert_cosine_diversity, so enabling both costs one encoder
            load, not two.
    Returns:
        Mean BERTScore distance over all N*(N-1)/2 unordered pairs; 0.0 if
        fewer than two outputs.
    """
    if len(outputs) < 2:
        return 0.0
    token_embeddings = _embed_tokens(outputs, model_name)
    distances = [
        _bertscore_distance(token_embeddings[i], token_embeddings[j])
        for i, j in combinations(range(len(outputs)), 2)
    ]
    return float(np.mean(distances))


def pairwise_distance(outputs: list[str]) -> float:
    """Naive string distance: mean pairwise absolute difference in length.

    Legacy Week 1 baseline (starter pack Known Issue #10), superseded by
    ast_node_diversity. Kept only so Week 2+ correlation plots can compare
    the structural metric against the naive one on identical runs. Two
    samples can have identical length while being logically unrelated, or
    be semantically identical at different lengths — length difference is
    at best a weak, indirect proxy for code distance.

    Args:
        outputs: List of N generated code strings for one problem.
    Returns:
        Mean |len(a) - len(b)| over all unordered pairs; 0.0 if fewer than
        two outputs. Unbounded above (character units, not normalized).
    """
    distances = [
        abs(len(outputs[i]) - len(outputs[j]))
        for i, j in combinations(range(len(outputs)), 2)
    ]
    return sum(distances) / len(distances) if distances else 0.0


def parse_failure_rate(outputs: list[str]) -> float:
    """Fraction of samples that fail to ast.parse() as valid Python.

    Not itself a diversity metric, but a cheap, well-defined diagnostic
    that ast_node_diversity also relies on internally (unparseable samples
    are maximally distant, per CLAUDE.md Section 6.2).

    Args:
        outputs: List of N generated code strings for one problem.
    Returns:
        count(unparseable) / N; 0.0 for an empty list.
    """
    if not outputs:
        return 0.0
    failures = sum(1 for code in outputs if _ast_node_sequence(code) is None)
    return failures / len(outputs)


# =============================================================================
# Week 7 — the prompt-inclusion confound: what string does a metric compare?
# =============================================================================
#
# run_pipeline.py assembles `prompt + sample` before executing (necessary: a
# bare HuggingFace completion is not standalone Python) and hands the *same*
# strings to compute_uncertainty(). Every diversity number recorded in Weeks
# 2-6 was therefore measured over strings whose prefix is identical across all
# N samples by construction — mean prompt share 0.695 (cg/HumanEval-164) and
# 0.731 (deepseek/HumanEval-60) by characters.
#
# That prefix is a confounder in both directions: it mechanically deflates
# measured diversity, and it independently predicts a *higher* pass rate
# (r = +0.289 / +0.458), so it induces a negative metric-vs-pass correlation
# with no self-consistency mechanism involved at all. Measured impact, over
# the stored runs: controlling for prompt share collapses
# token_jaccard_diversity from -0.219 to -0.032 at n=164, while recomputing
# it on completion-only strings lifts it to -0.548. ast_node_diversity
# (-0.580 -> -0.536) and lexical_diversity (-0.818 -> -0.800) barely move.
#
# The mechanism is dilution, and it is representation-specific: over the 1,569
# parseable stored samples of the merged-164 run the prompt is 0.695 of the
# string by characters and 0.693 by word tokens, but only 0.239 by AST nodes
# (a docstring is thousands of characters and a *single* Constant node), and a
# uniqueness count cannot be diluted at all, since a shared prefix can never
# make two distinct completions collide. That ordering — structural >
# lexical-uniqueness > token-overlap — is exactly the family robustness
# ordering the project has observed empirically since Week 2.
#
# The fix is per-family, because the families dilute differently and because a
# bare function body does not parse (stripping the prompt outright would make
# ast_node_diversity return the "unparseable => maximally distant" constant
# for every sample, i.e. undefined):
#
#   "full"       — Weeks 2-6 behavior, kept so every prior number reproduces.
#   "completion" — prompt removed entirely. Correct for the token/lexical and
#                  embedding families; makes the AST families undefined.
#   "signature"  — prompt minus its docstrings and comments, plus the
#                  completion. Keeps the code parseable while removing the
#                  prose mass that does the diluting. Correct for the
#                  structural families.
#   "corrected"  — the recommended configuration: each family gets the
#                  representation appropriate to it (token/lexical/embedding
#                  on "completion", structural on "signature").
#
# The execution-behavior family (execution_behavior_diversity, below) is
# immune by construction: it consumes per-probe outcome strings and never
# touches the source text at all.
#
# One measured side effect of "signature", found by re-running the full
# regime matrix and worth stating rather than burying: removing the prompt's
# docstring also removes the *only body statement* the prompt supplied, so a
# completion consisting purely of comments ("    # TODO: implement this")
# stops parsing, where prompt+completion parsed fine. It affected 7 of 60
# problems on the temperature-0.3 HumanEval run, and moved
# parse_failure_rate there from n.s. (rho -0.14) to FDR-significant (-0.31).
#
# That is arguably the correct behavior — an empty generation *is* a failed
# generation, and the docstring was masking it, which is the same
# prompt-dilution problem in another guise — but it does change what
# parse_failure_rate counts relative to Weeks 2-6, so any cross-week
# comparison of that one metric must say which representation it used.
#
# A stronger structural control exists and is deliberately not implemented
# here: re-attaching a synthetic `def _f():` header to the bare completion
# removes every shared prefix node rather than just the docstring. It is left
# out because completions that dedent to module level would become
# unparseable for reasons unrelated to model uncertainty.

METRIC_REPRESENTATIONS = ("full", "completion", "signature", "corrected")


def split_prompt_completion(full_code: str, prompt: str) -> tuple[str, str]:
    """Split an assembled candidate back into (prompt, model completion).

    Args:
        full_code: The stored `prompt + sample` string (run_pipeline.py).
        prompt: The problem prompt the sample was generated from.
    Returns:
        (prompt, completion) — the completion is the suffix the model
        actually produced.
    Raises:
        ValueError: If full_code does not start with prompt, which means the
            two do not belong together (wrong dataset, wrong task_id, or a
            prompt normalization change since the run was recorded). Failing
            loudly is deliberate: silently treating the whole string as a
            completion would corrupt every metric downstream.
    """
    if not full_code.startswith(prompt):
        raise ValueError(
            "stored code does not start with the given prompt "
            f"(code[:40]={full_code[:40]!r}, prompt[:40]={prompt[:40]!r})"
        )
    return prompt, full_code[len(prompt) :]


def strip_prompt_docstring(prompt: str) -> str:
    """Remove docstrings and comments from a prompt, keeping the code verbatim.

    Imports, decorators and signature lines survive character-for-character;
    only statement-initial string literals (docstrings) and `#` comments are
    removed. Operates on the token stream rather than on `ast`, because MBPP
    prompts end at the `def` line with no body and therefore do not parse at
    all, while they do tokenize.

    Args:
        prompt: The problem prompt.
    Returns:
        The prompt with docstrings/comments removed. Returned unchanged if
        the prompt cannot be tokenized (no silent corruption).
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(prompt).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        logger.debug("Prompt could not be tokenized; leaving it unstripped")
        return prompt

    lines = prompt.splitlines(keepends=True)
    drop_lines: set[int] = set()  # 1-based, dropped entirely
    truncate_at: dict[int, int] = {}  # 1-based line -> column to cut at

    # A statement-initial string literal is a docstring; the same literal in
    # the middle of an expression is data and must be kept.
    at_statement_start = True
    for token in tokens:
        if token.type == tokenize.COMMENT:
            before = lines[token.start[0] - 1][: token.start[1]]
            if before.strip():
                truncate_at[token.start[0]] = token.start[1]
            else:
                drop_lines.update(range(token.start[0], token.end[0] + 1))
        elif token.type == tokenize.STRING and at_statement_start:
            drop_lines.update(range(token.start[0], token.end[0] + 1))
            at_statement_start = False
        elif token.type in (
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENCODING,
        ):
            at_statement_start = True
        elif token.type != tokenize.COMMENT:
            at_statement_start = False

    kept: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        if lineno in drop_lines:
            continue
        if lineno in truncate_at:
            line = line[: truncate_at[lineno]].rstrip() + "\n"
        kept.append(line)
    return "".join(kept)


def prepare_metric_inputs(
    full_codes: list[str], prompt: str, *, representation: str = "full"
) -> list[str]:
    """Build the metric-input strings for one problem under a representation.

    Args:
        full_codes: The N stored `prompt + sample` strings for one problem.
        prompt: The problem prompt they were generated from.
        representation: "full", "completion" or "signature" (see this
            section's header comment). "corrected" is not a single string
            set — it is a per-family choice, so it is resolved by
            compute_uncertainty_prompt_aware() rather than here.
    Returns:
        One string per input sample, same order.
    Raises:
        ValueError: On an unknown representation, or (via
            split_prompt_completion) if a sample does not start with prompt.
    """
    if representation == "full":
        return list(full_codes)
    if representation == "completion":
        return [split_prompt_completion(code, prompt)[1] for code in full_codes]
    if representation == "signature":
        stub = strip_prompt_docstring(prompt)
        return [stub + split_prompt_completion(code, prompt)[1] for code in full_codes]
    raise ValueError(
        f"unknown representation {representation!r}; "
        f"expected one of {METRIC_REPRESENTATIONS}"
    )


def _node_count(code: str) -> int | None:
    """AST node count, or None if the code does not parse."""
    sequence = _ast_node_sequence(code)
    return None if sequence is None else len(sequence)


def prompt_confound_covariates(full_codes: list[str], prompt: str) -> dict[str, float]:
    """Per-problem covariates for the two confounds of Week 7's re-analysis.

    Two distinct quantities, needing two distinct treatments:

      - prompt_share_* is a *measurement artefact* — the fraction of the
        compared string that is identical prompt. The fix is to recompute on
        a prompt-free representation (see prepare_metric_inputs), and this
        covariate exists to quantify what the uncorrected numbers were
        measuring.
      - completion/prompt *length* is a *substantive* confound — longer
        solutions are genuinely harder and genuinely more varied, so length
        drives both the metric and the label. It must be controlled for and
        reported as a partial correlation, never stripped. It also serves as
        a free baseline: across all six of this project's regime cells, mean
        completion length alone recovers 71-96% of ast_node_diversity's
        correlation magnitude with pass rate.

    The AST share is reported because dilution is representation-specific:
    the prompt is ~0.69 of the string by characters and by word tokens, but
    only ~0.24 by AST nodes, which is why the token family is hit hardest.

    Args:
        full_codes: The N stored `prompt + sample` strings for one problem.
        prompt: The problem prompt they were generated from.
    Returns:
        Dict with mean_completion_chars, mean_completion_tokens,
        prompt_chars, prompt_tokens, prompt_share_char, prompt_share_token
        and prompt_share_ast (float("nan") when the AST share cannot be
        computed because a sample or the prompt does not parse).
    """
    if not full_codes:
        return {}
    completions = [split_prompt_completion(code, prompt)[1] for code in full_codes]
    prompt_chars = len(prompt)
    prompt_tokens = len(_WORD_TOKEN_RE.findall(prompt))
    completion_chars = [len(c) for c in completions]
    completion_tokens = [len(_WORD_TOKEN_RE.findall(c)) for c in completions]

    # MBPP prompts stop at the `def` line and do not parse on their own; a
    # single `pass` body makes them parseable, and its one node is subtracted
    # back out so the count means "nodes the prompt contributes".
    prompt_nodes = _node_count(prompt)
    if prompt_nodes is None:
        with_body = _node_count(prompt + "    pass\n")
        prompt_nodes = None if with_body is None else with_body - 1

    full_nodes = [_node_count(code) for code in full_codes]
    parseable = [n for n in full_nodes if n]
    if prompt_nodes is None or not parseable:
        share_ast = float("nan")
    else:
        share_ast = float(np.mean([prompt_nodes / n for n in parseable]))

    mean_completion_chars = float(np.mean(completion_chars))
    mean_completion_tokens = float(np.mean(completion_tokens))
    return {
        "mean_completion_chars": mean_completion_chars,
        "mean_completion_tokens": mean_completion_tokens,
        "prompt_chars": float(prompt_chars),
        "prompt_tokens": float(prompt_tokens),
        "prompt_share_char": float(
            np.mean([prompt_chars / (prompt_chars + n) for n in completion_chars])
        ),
        "prompt_share_token": float(
            np.mean(
                [
                    prompt_tokens / (prompt_tokens + n) if prompt_tokens + n else 0.0
                    for n in completion_tokens
                ]
            )
        ),
        "prompt_share_ast": share_ast,
    }


def _compute_uncertainty_split(
    lexical_outputs: list[str],
    structural_outputs: list[str],
    *,
    include_codebert: bool = False,
    include_tree_edit: bool = True,
    codebert_model_name: str = DEFAULT_CODEBERT_MODEL,
) -> dict[str, float]:
    """Single source of truth for the metric dict, over per-family inputs.

    Both public entry points delegate here: compute_uncertainty() passes the
    same list twice (Weeks 2-6 behavior, unchanged), while
    compute_uncertainty_prompt_aware() can pass a different representation to
    each family. Adding a metric here adds it to both.

    Private helper: the family assignment of each metric — which is a
    research decision, not an implementation detail — is documented in this
    section's header comment.
    """
    metrics = {
        "lexical_diversity": lexical_diversity(lexical_outputs),
        "token_jaccard_diversity": token_jaccard_diversity(lexical_outputs),
        "token_cosine_diversity": token_cosine_diversity(lexical_outputs),
        "ast_node_diversity": ast_node_diversity(structural_outputs),
        "pairwise_distance": pairwise_distance(lexical_outputs),
        "parse_failure_rate": parse_failure_rate(structural_outputs),
    }
    if include_tree_edit:
        metrics["tree_edit_distance_diversity"] = tree_edit_distance_diversity(
            structural_outputs
        )
    if include_codebert:
        metrics["codebert_cosine_diversity"] = codebert_cosine_diversity(
            lexical_outputs, model_name=codebert_model_name
        )
        metrics["codebert_bertscore_diversity"] = codebert_bertscore_diversity(
            lexical_outputs, model_name=codebert_model_name
        )
    return metrics


def compute_uncertainty_prompt_aware(
    full_codes: list[str],
    prompt: str,
    *,
    representation: str = "corrected",
    include_codebert: bool = False,
    include_tree_edit: bool = True,
    codebert_model_name: str = DEFAULT_CODEBERT_MODEL,
) -> dict[str, float]:
    """compute_uncertainty() with the prompt-inclusion confound controlled.

    Same metric keys and same value ranges as compute_uncertainty(); the
    only difference is which string each metric family is computed over.
    representation="full" reproduces compute_uncertainty() exactly, so the
    two can be compared on identical stored runs.

    Args:
        full_codes: The N stored `prompt + sample` strings for one problem.
        prompt: The problem prompt they were generated from.
        representation: One of METRIC_REPRESENTATIONS. Default "corrected":
            token/lexical/embedding metrics on completion-only strings,
            structural metrics on docstring-stripped signature + completion.
        include_codebert: As in compute_uncertainty().
        include_tree_edit: As in compute_uncertainty().
        codebert_model_name: As in compute_uncertainty().
    Returns:
        The same dict compute_uncertainty() returns.
    Raises:
        ValueError: On an unknown representation, or if a sample does not
            start with prompt.
    """
    if representation == "corrected":
        lexical_inputs = prepare_metric_inputs(
            full_codes, prompt, representation="completion"
        )
        structural_inputs = prepare_metric_inputs(
            full_codes, prompt, representation="signature"
        )
    else:
        lexical_inputs = prepare_metric_inputs(
            full_codes, prompt, representation=representation
        )
        structural_inputs = lexical_inputs
    return _compute_uncertainty_split(
        lexical_inputs,
        structural_inputs,
        include_codebert=include_codebert,
        include_tree_edit=include_tree_edit,
        codebert_model_name=codebert_model_name,
    )


def compute_uncertainty(
    outputs: list[str],
    *,
    include_codebert: bool = False,
    include_tree_edit: bool = True,
    codebert_model_name: str = DEFAULT_CODEBERT_MODEL,
) -> dict[str, float]:
    """Compute all implemented uncertainty metrics for a set of samples.

    Default set: the three Week 2 contract metrics (lexical_diversity,
    token_jaccard_diversity, ast_node_diversity) plus the literature-
    grounded token_cosine_diversity (same lexical family as
    token_jaccard_diversity, cosine instead of Jaccard — see its
    docstring), the Week 3 tree_edit_distance_diversity, the legacy Week 1
    baseline pairwise_distance, and the parse_failure_rate diagnostic.
    Opt-in (keyword-gated, both load the same cached CodeBERT encoder):
    codebert_cosine_diversity (Week 3, mean-pooled sentence embedding) and
    codebert_bertscore_diversity (greedy per-token alignment — see its
    docstring). Both CodeBERT metrics are gated together because they
    share one ~500 MB encoder load: entry points wanting the full set
    (run_pipeline.py, analyze_uncertainty.py) pass include_codebert=True,
    while unit tests and cheap callers stay offline by default. Callers
    should treat this dict's key set as growing, not fixed (Week 4+ adds
    more).

    Args:
        outputs: List of N generated code strings for one problem. Callers
            should pass fully-assembled candidate code (prompt + sample),
            not the raw model completion alone — a bare completion is
            usually not valid standalone Python (see run_pipeline.py),
            which would make parse_failure_rate and the AST-based metrics
            meaningless.
        include_codebert: If True, also compute codebert_cosine_diversity
            and codebert_bertscore_diversity (downloads/loads the encoder
            on the first call).
        include_tree_edit: If False, omit tree_edit_distance_diversity.
            Default True, so every pre-Week-7 caller is unchanged. It is a
            separate gate from include_codebert because it is expensive for a
            different reason: Zhang-Shasha is superlinear in tree size and the
            pair count is O(N^2), so at N=50 it measures ~99 s per problem
            against ~0.3 s for every other metric combined. Turn it off for
            large-N generation runs whose analysis does not read it (the
            reliability and N-sweep scripts exclude it by default).
        codebert_model_name: HuggingFace encoder for both CodeBERT metrics.
    Returns:
        Dict with keys "lexical_diversity", "token_jaccard_diversity",
        "token_cosine_diversity", "ast_node_diversity",
        "tree_edit_distance_diversity", "pairwise_distance",
        "parse_failure_rate", and (only when enabled)
        "codebert_cosine_diversity", "codebert_bertscore_diversity".

    Note (Week 7): passing assembled prompt+sample strings means every metric
    compares strings that share an identical prefix, which measurably
    confounds the token/lexical families — see
    compute_uncertainty_prompt_aware() and this file's "prompt-inclusion
    confound" section. This function's behavior is unchanged so that every
    Week 2-6 number reproduces exactly.
    """
    return _compute_uncertainty_split(
        outputs,
        outputs,
        include_codebert=include_codebert,
        include_tree_edit=include_tree_edit,
        codebert_model_name=codebert_model_name,
    )


# =============================================================================
# Week 4 — per-sample self-consistency features (SelfCheckGPT-style)
# =============================================================================


def _row_means_excluding_diagonal(matrix: np.ndarray) -> list[float]:
    """Per-row mean of a symmetric NxN distance matrix with a zero diagonal.

    Row i's sum already excludes self-distance (the diagonal is 0), so
    dividing by n - 1 gives sample i's mean distance to the other n - 1
    samples directly. n <= 1 returns 0.0 for every row — a single sample
    cannot disagree with itself, the same convention every aggregate
    diversity metric in this file uses for < 2 outputs.
    """
    n = matrix.shape[0]
    if n <= 1:
        return [0.0] * n
    return [float(v) for v in matrix.sum(axis=1) / (n - 1)]


def token_jaccard_self_consistency(outputs: list[str]) -> list[float]:
    """Per-sample leave-one-out mean token-Jaccard distance.

    Sample i's score is its mean Jaccard distance (token_jaccard_diversity's
    pairwise term) to the other N - 1 samples of the same problem — the
    lexical/presence-absence self-consistency signal (SelfCheckGPT,
    Manakul et al., 2023, EMNLP). Range [0.0, 1.0] per sample; higher = this
    specific sample disagrees more with its siblings.

    Args:
        outputs: List of N generated code strings for one problem.
    Returns:
        List of N per-sample scores, same order as outputs.
    """
    n = len(outputs)
    if n <= 1:
        return [0.0] * n
    token_sets = [_word_token_set(code) for code in outputs]
    matrix = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        d = _jaccard_distance(token_sets[i], token_sets[j])
        matrix[i, j] = matrix[j, i] = d
    return _row_means_excluding_diagonal(matrix)


def token_cosine_self_consistency(outputs: list[str]) -> list[float]:
    """Per-sample leave-one-out mean token-count cosine distance.

    Same idea as token_jaccard_self_consistency, using
    token_cosine_diversity's frequency-weighted pairwise term instead of
    Jaccard's presence/absence one. Range [0.0, 1.0] per sample.

    Args:
        outputs: List of N generated code strings for one problem.
    Returns:
        List of N per-sample scores, same order as outputs.
    """
    n = len(outputs)
    if n <= 1:
        return [0.0] * n
    counts = [_word_token_counts(code) for code in outputs]
    matrix = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        d = _cosine_distance_from_counts(counts[i], counts[j])
        matrix[i, j] = matrix[j, i] = d
    return _row_means_excluding_diagonal(matrix)


def ast_node_self_consistency(outputs: list[str]) -> list[float]:
    """Per-sample leave-one-out mean AST node-sequence distance.

    Same idea as token_jaccard_self_consistency, using ast_node_diversity's
    normalized-Levenshtein pairwise term — the structural self-consistency
    signal. Unparseable samples keep ast_node_diversity's contract: distance
    1.0 to every other sample (parseable or not). Range [0.0, 1.0] per
    sample.

    Tree edit distance's per-sample variant is deliberately not offered:
    the Week 3 addendum's redundancy heatmap found tree_edit_distance_diversity
    and ast_node_diversity rank-correlated at rho=+0.99 at the aggregate
    level, and TED is ~339x more expensive per problem (Week 3 report,
    Section 6); paying that cost N times per problem (once per leave-one-out
    row) for a near-duplicate per-sample signal is not worth it.

    Args:
        outputs: List of N generated code strings for one problem.
    Returns:
        List of N per-sample scores, same order as outputs.
    """
    n = len(outputs)
    if n <= 1:
        return [0.0] * n
    sequences = [_ast_node_sequence(code) for code in outputs]
    matrix = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        if sequences[i] is None or sequences[j] is None:
            d = 1.0
        else:
            d = _normalized_levenshtein(sequences[i], sequences[j])
        matrix[i, j] = matrix[j, i] = d
    return _row_means_excluding_diagonal(matrix)


def codebert_cosine_self_consistency(
    outputs: list[str], model_name: str = DEFAULT_CODEBERT_MODEL
) -> list[float]:
    """Per-sample leave-one-out mean CodeBERT cosine distance.

    Same idea as token_jaccard_self_consistency, using
    codebert_cosine_diversity's mean-pooled-embedding pairwise term — the
    semantic self-consistency signal, and (per codebert_cosine_diversity's
    own docstring) the only family here that still yields signal on
    unparseable samples. BERTScore's per-token variant is not offered here
    for the same cost/redundancy reasoning as tree edit distance above
    (Week 3 addendum Observation 7: codebert_bertscore_diversity correlates
    rho=0.87-0.88 with the structural family, the highest redundancy of any
    metric measured).

    Args:
        outputs: List of N generated code strings for one problem.
        model_name: HuggingFace encoder identifier; shares _load_codebert's
            cache with the other CodeBERT metrics.
    Returns:
        List of N per-sample scores, same order as outputs.
    """
    n = len(outputs)
    if n <= 1:
        return [0.0] * n
    embeddings = _embed_outputs(outputs, model_name)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.clip(norms, 1e-12, None)
    similarity = unit @ unit.T
    matrix = 1.0 - similarity
    np.fill_diagonal(matrix, 0.0)
    return _row_means_excluding_diagonal(matrix)


def compute_self_consistency(
    outputs: list[str],
    *,
    include_codebert: bool = False,
    codebert_model_name: str = DEFAULT_CODEBERT_MODEL,
) -> list[dict[str, float]]:
    """Per-sample self-consistency features, one dict per sample.

    The sample-level counterpart to compute_uncertainty()'s per-problem
    dict: where compute_uncertainty() answers "how much do this problem's N
    samples disagree overall," this answers "how much does *this specific*
    sample disagree with its siblings" — a genuinely per-sample feature
    vector, which a classifier needs to tell samples of the same problem
    apart (see the Week 4 module-docstring section above for why the
    problem-level metrics cannot do this).

    Args:
        outputs: List of N generated code strings for one problem.
        include_codebert: If True, also compute
            codebert_cosine_self_consistency (loads the ~500 MB encoder on
            first use; shares the cache with compute_uncertainty()'s
            CodeBERT metrics).
        codebert_model_name: HuggingFace encoder for the CodeBERT feature.
    Returns:
        List of N dicts (same order as outputs), each with keys
        "token_jaccard_self_distance", "token_cosine_self_distance",
        "ast_node_self_distance", and (only when enabled)
        "codebert_cosine_self_distance".
    """
    jaccard_scores = token_jaccard_self_consistency(outputs)
    cosine_scores = token_cosine_self_consistency(outputs)
    ast_scores = ast_node_self_consistency(outputs)
    per_sample = [
        {
            "token_jaccard_self_distance": jaccard_scores[i],
            "token_cosine_self_distance": cosine_scores[i],
            "ast_node_self_distance": ast_scores[i],
        }
        for i in range(len(outputs))
    ]
    if include_codebert:
        codebert_scores = codebert_cosine_self_consistency(
            outputs, model_name=codebert_model_name
        )
        for row, score in zip(per_sample, codebert_scores):
            row["codebert_cosine_self_distance"] = score
    return per_sample


def compute_self_consistency_prompt_aware(
    full_codes: list[str],
    prompt: str,
    *,
    representation: str = "corrected",
    include_codebert: bool = False,
    codebert_model_name: str = DEFAULT_CODEBERT_MODEL,
) -> list[dict[str, float]]:
    """compute_self_consistency() with the prompt-inclusion confound controlled.

    The per-sample features are built from the same pairwise distance terms
    as the problem-level metrics, so they inherit the same confound: a
    shared prompt prefix dilutes the token families far more than the
    structural one (see this file's "prompt-inclusion confound" section).
    Task B's headline predictor numbers are built on these features, so
    correcting them matters at least as much as correcting the
    problem-level table.

    Args:
        full_codes: The N stored `prompt + sample` strings for one problem.
        prompt: The problem prompt they were generated from.
        representation: One of METRIC_REPRESENTATIONS. "full" reproduces
            compute_self_consistency() exactly.
        include_codebert: As in compute_self_consistency().
        codebert_model_name: As in compute_self_consistency().
    Returns:
        The same list of per-sample dicts compute_self_consistency() returns.
    Raises:
        ValueError: On an unknown representation, or if a sample does not
            start with prompt.
    """
    if representation == "corrected":
        lexical_inputs = prepare_metric_inputs(
            full_codes, prompt, representation="completion"
        )
        structural_inputs = prepare_metric_inputs(
            full_codes, prompt, representation="signature"
        )
    else:
        lexical_inputs = prepare_metric_inputs(
            full_codes, prompt, representation=representation
        )
        structural_inputs = lexical_inputs

    jaccard_scores = token_jaccard_self_consistency(lexical_inputs)
    cosine_scores = token_cosine_self_consistency(lexical_inputs)
    ast_scores = ast_node_self_consistency(structural_inputs)
    per_sample = [
        {
            "token_jaccard_self_distance": jaccard_scores[i],
            "token_cosine_self_distance": cosine_scores[i],
            "ast_node_self_distance": ast_scores[i],
        }
        for i in range(len(full_codes))
    ]
    if include_codebert:
        codebert_scores = codebert_cosine_self_consistency(
            lexical_inputs, model_name=codebert_model_name
        )
        for row, score in zip(per_sample, codebert_scores):
            row["codebert_cosine_self_distance"] = score
    return per_sample


# =============================================================================
# Week 6 — Execution-behavior diversity (the functional/behavioral axis)
# =============================================================================
#
# Every metric above compares samples as text, structure, or embeddings.
# None observes what the code *does* when run — and four independent
# 2025-2026 papers converge on exactly that gap in sample-based code
# uncertainty estimation:
#   - He, Sharma & David (arXiv 2605.09023): "semantic distance-aware"
#     uncertainty — how severely execution behaviors differ — beats textual
#     baselines across four benchmarks.
#   - "Code Is More Than Text" (arXiv 2606.09577): a functional/behavioral-
#     consistency axis alongside lexical and algorithmic ones (+8.1 AUROC
#     over the strongest NL-derived baseline).
#   - "Functional Entropy" (arXiv 2605.28500): NLI/embedding-similarity
#     methods fail on code precisely because they cannot distinguish
#     functionally different programs; functional-equivalence methods win.
#   - DiffCodeGen (Yifeng He, Wang, Wang, Ouyang & Chen, arXiv 2605.20473):
#     the operational recipe — execute all candidates on shared inputs,
#     cluster by behavioral similarity, return the largest cluster's medoid.
#     NOTE: a different first author and group from the Weilin He of
#     2605.09023 above; the two are frequently conflated because they share
#     a surname and a month.
#
# None of the four decomposes its execution signal into "does this program
# run at all" and "does it agree with the others where it does run", and
# none reports the signal restricted to running programs — the gap the
# crash-rate control in the Week 7 report Section 7.3 fills. Crash pollution
# is not a small-model artefact: 2605.09023 App. A reports 0.201-0.269 of
# execution cells crashing on frontier models, against 0.134-0.373 measured
# here on the same denominator.
#
# Design constraint (leakage/tautology): these metrics compare samples'
# *outputs on shared probe inputs* (evaluation.behavior_signature), never
# test verdicts. Two samples that both fail the tests but fail *identically*
# score 0 distance; pass/fail agreement is deliberately not a feature — a
# problem where all N samples fail has "all agree they fail" written into
# the Task A label itself, so verdict agreement would partially reconstruct
# the target rather than measure behavior.
#
# The probe/signature machinery lives in src/evaluation.py (all execution
# stays there, same # TODO: SANDBOX caveat); these functions consume the
# precomputed signatures and stay execution-free, like every other metric
# in this file.


# Graded per-probe outcome distances for abnormal executions, following
# He et al. (arXiv 2605.09023, Eq. 2): (a, b, c) = cost when exactly one
# outcome is abnormal / both abnormal with different error types / both
# abnormal with the SAME error type. Their default (1.0, 0.8, 0.6) reflects
# that "identical errors may indicate related failure modes, but need not
# imply behavioural equivalence"; their sensitivity analysis puts fixed
# weights within 0.011 AUROC of per-setting learned ones. The Week 6
# overnight results used the (1.0, 1.0, 0.0) corner (same-error = full
# agreement) — keep that reachable via error_costs for continuity arms.
DEFAULT_ERROR_COSTS = (1.0, 0.8, 0.6)
LEGACY_ERROR_COSTS = (1.0, 1.0, 0.0)


def _outcome_distance(x: str, y: str, error_costs: tuple[float, float, float]) -> float:
    # Per-probe graded distance; "abnormal" = behavior_signature's "error:*".
    x_err, y_err = x.startswith("error:"), y.startswith("error:")
    if not x_err and not y_err:
        return 0.0 if x == y else 1.0
    a, b, c = error_costs
    if x_err != y_err:
        return a
    return c if x == y else b


def _signature_distance(
    a: list[str],
    b: list[str],
    error_costs: tuple[float, float, float] = DEFAULT_ERROR_COSTS,
) -> float:
    """Mean graded per-probe outcome distance between two behavior signatures."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(_outcome_distance(x, y, error_costs) for x, y in zip(a, b)) / n


def execution_behavior_diversity(
    signatures: list[list[str]],
    error_costs: tuple[float, float, float] = DEFAULT_ERROR_COSTS,
) -> float:
    """Average pairwise behavioral disagreement across N samples.

    Each signature is one sample's per-probe outcome list from
    evaluation.behavior_signature(); a pair's distance is the mean graded
    per-probe outcome distance (equal normal outputs = 0; different normal
    outputs = 1; abnormal outcomes graded by error_costs — see
    DEFAULT_ERROR_COSTS and He et al., arXiv 2605.09023, Eq. 2), always
    independent of test verdicts. Higher = the samples *behave* more
    differently when run = higher functional uncertainty. See the
    module-section comment above for the four grounding papers.

    Args:
        signatures: One outcome list per sample, all over the same probes.
        error_costs: (a, b, c) graded abnormal-outcome costs; pass
            LEGACY_ERROR_COSTS for the pre-graded Week 6 overnight behavior.
    Returns:
        Mean pairwise distance over all N(N-1)/2 pairs, in [0.0, 1.0];
        0.0 for fewer than two samples or an empty probe set.
    """
    if len(signatures) < 2:
        return 0.0
    distances = [
        _signature_distance(a, b, error_costs) for a, b in combinations(signatures, 2)
    ]
    return float(np.mean(distances)) if distances else 0.0


def execution_behavior_self_consistency(
    signatures: list[list[str]],
    error_costs: tuple[float, float, float] = DEFAULT_ERROR_COSTS,
) -> list[float]:
    """Per-sample leave-one-out behavioral disagreement (Task B feature).

    Sample i's score is its mean signature distance to the other N-1
    samples — the execution-behavior member of the compute_self_consistency
    feature family (SelfCheckGPT mechanism, applied to behavior instead of
    text/structure/embeddings). Verdict-free by construction (see the
    module-section comment), so it is safe as a sample-level feature: it
    never reads the sample's own pass/fail label. Distances use the same
    graded error_costs as execution_behavior_diversity.

    Args:
        signatures: One outcome list per sample, all over the same probes.
        error_costs: (a, b, c) graded abnormal-outcome costs.
    Returns:
        List of N leave-one-out mean distances; all 0.0 when N < 2.
    """
    n = len(signatures)
    if n < 2:
        return [0.0] * n
    scores = []
    for i in range(n):
        others = [
            _signature_distance(signatures[i], s, error_costs)
            for j, s in enumerate(signatures)
            if j != i
        ]
        scores.append(float(np.mean(others)))
    return scores


# =============================================================================
# Week 7 — the crash-rate control for the execution-behavior family
# =============================================================================
#
# Week 6's headline is that the execution-behavior features are the project's
# strongest signal (Task B ROC-AUC 0.89-0.93). There is a duller explanation
# that has never been ruled out: at 350M, parse-failure rates reach 0.7 and
# 18% of the original MBPP probe outcomes were "error:unrunnable", so
# "behavioral consensus predicts correctness" may partly be "*runnability*
# predicts correctness" — a fact about whether the generated code runs at all,
# not about whether the samples agree on what it computes. The 7B+ papers
# implicitly control for this, because their models rarely emit unrunnable
# code; this project cannot assume it.
#
# Two decompositions, both verdict-free like everything else in this family:
#
#   execution_crash_rate           — the *duller* half on its own: what
#                                    fraction of probes does this sample
#                                    error on? No agreement notion at all.
#   execution_behavior_*_nocrash   — the *interesting* half on its own:
#                                    behavioral disagreement restricted to
#                                    probes where BOTH samples ran normally.
#
# The Week 6 claim is safe, and strengthened, iff the nocrash variant keeps
# most of the ROC-AUC and clearly beats crash rate alone. If instead crash
# rate carries it, the honest framing changes from "behavioral consensus" to
# "runnability", which the project needs to know before publication rather
# than from a reviewer.


def execution_crash_rate(signature: list[str]) -> float:
    """Fraction of probes on which one sample produced an abnormal outcome.

    "Abnormal" is behavior_signature()'s "error:*" convention: a raised
    exception, a timeout, or an unrunnable sample. Deliberately not a
    diversity metric — it is the single-sample confound that the
    execution-behavior family has to be shown to beat.

    Args:
        signature: One sample's per-probe outcome list.
    Returns:
        count(abnormal) / n_probes; 0.0 for an empty probe set.
    """
    if not signature:
        return 0.0
    return sum(1 for outcome in signature if outcome.startswith("error:")) / len(
        signature
    )


def _signature_distance_nocrash(a: list[str], b: list[str]) -> float:
    """Pairwise distance over probes where BOTH samples ran normally.

    Probes on which either sample errored are dropped rather than graded,
    so the result cannot be driven by runnability. Returns 0.0 when no
    probe survives, matching this family's "no evidence of disagreement =
    0.0" convention (an empty probe set already behaves that way) — which
    is conservative here, since a pair that never co-ran is scored as
    agreeing rather than as maximally distant.
    """
    surviving = [
        (x, y)
        for x, y in zip(a, b)
        if not x.startswith("error:") and not y.startswith("error:")
    ]
    if not surviving:
        return 0.0
    return sum(0.0 if x == y else 1.0 for x, y in surviving) / len(surviving)


def execution_behavior_diversity_nocrash(signatures: list[list[str]]) -> float:
    """execution_behavior_diversity restricted to probes both samples ran.

    Args:
        signatures: One outcome list per sample, all over the same probes.
    Returns:
        Mean pairwise distance in [0.0, 1.0]; 0.0 for fewer than two
        samples, an empty probe set, or no co-running probe.
    """
    if len(signatures) < 2:
        return 0.0
    distances = [
        _signature_distance_nocrash(a, b) for a, b in combinations(signatures, 2)
    ]
    return float(np.mean(distances)) if distances else 0.0


def execution_behavior_self_consistency_nocrash(
    signatures: list[list[str]],
) -> list[float]:
    """Per-sample leave-one-out counterpart of the nocrash diversity.

    Args:
        signatures: One outcome list per sample, all over the same probes.
    Returns:
        List of N leave-one-out mean distances; all 0.0 when N < 2.
    """
    n = len(signatures)
    if n < 2:
        return [0.0] * n
    return [
        float(
            np.mean(
                [
                    _signature_distance_nocrash(signatures[i], s)
                    for j, s in enumerate(signatures)
                    if j != i
                ]
            )
        )
        for i in range(n)
    ]


# =============================================================================
# Week 4 — UncertaintyPredictor (CLAUDE.md Section 10.1)
# =============================================================================


class UncertaintyPredictor:
    """Binary pass/fail classifier over uncertainty-metric features.

    Thin wrapper around an sklearn Pipeline(StandardScaler, model) so both
    supported model types share one fit/predict/predict_proba surface and
    one preprocessing step. StandardScaler is applied even for the
    random-forest model: a per-feature affine rescaling cannot change a
    decision tree's split points or feature importances (trees split on
    order, not magnitude), so it is a no-op for that model rather than a
    biasing step — keeping one pipeline shape for both models is simpler
    than branching, not a modeling compromise.

    Model choice (CLAUDE.md Week 4 plan: "logistic regression + a
    tree-based baseline"):
      - "logistic": LogisticRegression(class_weight="balanced"). The linear,
        interpretable baseline; coefficients on standardized features are
        directly comparable feature-importance signs/magnitudes.
      - "random_forest": RandomForestClassifier(class_weight="balanced").
        Non-linear, handles the Week 3 addendum's documented multicollinearity
        (e.g. tree_edit_distance_diversity vs ast_node_diversity, rho=0.99)
        without the coefficient instability a collinear linear model gets.

    class_weight="balanced" on both models: every dataset used here is
    imbalanced (this project's runs sit around 20-25% passing), and an
    unweighted classifier can reach high accuracy by mostly predicting the
    majority class — balancing re-weights the loss so both classes matter
    during fitting, independent of how the resulting predictions are scored.

    Args:
        model_type: "logistic" or "random_forest".
        seed: Random seed for the model's internal randomness (CLAUDE.md
            Constraint #4). Reproducibility only; not a diversity metric.
    """

    _MODEL_TYPES = ("logistic", "random_forest")

    def __init__(self, model_type: str = "logistic", seed: int = 42) -> None:
        if model_type not in self._MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type {model_type!r}; choose from {self._MODEL_TYPES}"
            )
        self.model_type = model_type
        self.seed = seed
        self.feature_names_: list[str] | None = None
        if model_type == "logistic":
            model = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=seed
            )
        else:
            model = RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
        self._pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None
    ) -> "UncertaintyPredictor":
        """Fit the pipeline. feature_names (optional) label feature_importances()."""
        self.feature_names_ = (
            list(feature_names)
            if feature_names
            else [f"x{i}" for i in range(X.shape[1])]
        )
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class labels (0/1), shape (n_samples,)."""
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicted probability of the positive (pass) class, shape (n_samples,)."""
        return self._pipeline.predict_proba(X)[:, 1]

    def feature_importances(self) -> dict[str, float]:
        """Per-feature signal: standardized logistic coefficients, or
        random-forest impurity importances. Not comparable across model
        types (signed vs. non-negative, different units) — compare within
        one model_type only.
        """
        if self.feature_names_ is None:
            raise RuntimeError("Call fit() before feature_importances().")
        model = self._pipeline.named_steps["model"]
        if self.model_type == "logistic":
            values = model.coef_[0]
        else:
            values = model.feature_importances_
        return dict(zip(self.feature_names_, (float(v) for v in values)))


def evaluate_predictor_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    model_type: str = "logistic",
    n_splits: int = 5,
    seed: int = 42,
    feature_names: list[str] | None = None,
) -> dict:
    """Leave-problems-out cross-validated evaluation of UncertaintyPredictor.

    Uses StratifiedGroupKFold so that (a) no problem's rows appear in both
    the train and test side of any fold — required because rows from the
    same problem are not independent (the Week 2/3 reports repeatedly flag
    this for point-biserial correlation; here it would be outright *label
    leakage* for a plain KFold, since same-problem rows can share nearly
    identical feature values) — and (b) the pass/fail ratio is preserved
    per fold despite class imbalance, so no fold degenerates to (near-)one
    class. n_splits is capped at the number of distinct groups, since
    StratifiedGroupKFold cannot produce more folds than there are groups.

    Two metric families are reported per fold: precision/recall/F1 on the
    positive (pass) class, and their macro-averaged counterparts (equal
    weight per class, not per sample — informative on its own given the
    imbalance). ROC-AUC and average precision (PR-AUC) are also reported;
    PR-AUC is included specifically because ROC-AUC can look deceptively
    strong under class imbalance while precision-recall does not (Davis &
    Goadrich, 2006, "The Relationship Between Precision-Recall and ROC
    Curves," ICML) — both are None for a fold whose test split has only one
    class present (undefined for either curve).

    Args:
        X: Feature matrix, shape (n_rows, n_features).
        y: Binary labels (0/1), shape (n_rows,).
        groups: Group id per row (e.g. task_id), shape (n_rows,). Rows
            sharing a group id never split across train/test.
        model_type: "logistic" or "random_forest" (see UncertaintyPredictor).
        n_splits: Requested fold count; capped at the number of groups.
        seed: Random seed forwarded to UncertaintyPredictor and used to
            shuffle fold assignment (CLAUDE.md Constraint #4).
        feature_names: Optional feature labels for the returned
            feature_importances (fit once on the full data after CV scoring
            — CV folds only ever produce held-out predictions, never a
            model this function returns).
    Returns:
        Dict with config, per-fold metrics, mean +/- std summary, and
        feature_importances from a final fit on all rows.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))

    # Out-of-fold predictions: every row gets exactly one prediction, made by
    # a model that never saw that row's group during fitting. This is what
    # Week 5's calibration analysis (compute_calibration) needs -- an
    # in-sample predict_proba() (fit and score on the same rows) is
    # systematically overconfident/well-calibrated by construction, since
    # the model was optimized against those exact labels.
    oof_proba = np.full(len(y), np.nan)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []
    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        predictor = UncertaintyPredictor(model_type=model_type, seed=seed)
        predictor.fit(X[train_idx], y[train_idx], feature_names=feature_names)
        y_test = y[test_idx]
        y_pred = predictor.predict(X[test_idx])
        y_proba = predictor.predict_proba(X[test_idx])
        oof_proba[test_idx] = y_proba

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        if len(np.unique(y_test)) == 2:
            roc_auc: float | None = float(roc_auc_score(y_test, y_proba))
            pr_auc: float | None = float(average_precision_score(y_test, y_proba))
        else:
            roc_auc = None
            pr_auc = None
        fold_metrics.append(
            {
                "fold": fold_i,
                "n_test": int(len(test_idx)),
                "n_test_groups": int(len(np.unique(groups[test_idx]))),
                "n_test_positive": int(y_test.sum()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

    summary = {}
    for key in (
        "precision",
        "recall",
        "f1",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc",
        "pr_auc",
    ):
        values = [m[key] for m in fold_metrics if m[key] is not None]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else None
        summary[f"{key}_std"] = float(np.std(values)) if values else None

    final_model = UncertaintyPredictor(model_type=model_type, seed=seed).fit(
        X, y, feature_names=feature_names
    )
    return {
        "model_type": model_type,
        "n_splits": n_splits,
        "n_features": int(X.shape[1]),
        "feature_names": list(feature_names) if feature_names else None,
        "n_rows": int(len(y)),
        "n_groups": n_groups,
        "class_balance": {"pass": int(y.sum()), "fail": int(len(y) - int(y.sum()))},
        "fold_metrics": fold_metrics,
        "summary": summary,
        "feature_importances": final_model.feature_importances(),
        # Every row's held-out probability (see the oof_proba comment above)
        # plus the labels in the same order, so a caller (e.g. Week 5's
        # compute_calibration) can score calibration without re-deriving
        # which fold predicted which row.
        "oof_proba": oof_proba.tolist(),
        "oof_y": y.tolist(),
    }


# =============================================================================
# Week 5 — Calibration (CLAUDE.md Section 10.1: "Expected Calibration Error,
# reliability diagram")
# =============================================================================


def compute_calibration(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    *,
    n_bins: int | None = None,
    strategy: str = "quantile",
) -> dict:
    """Expected Calibration Error, reliability-diagram bins, and Brier score.

    Answers a different question than every prior metric in this file: not
    "does this uncertainty signal correlate with pass/fail" (Weeks 2-4), but
    "when UncertaintyPredictor.predict_proba() says 70%, does the sample
    actually pass about 70% of the time" -- calibration, not just ranking
    (a well-ranked score can still be miscalibrated, e.g. uniformly too
    confident). y_prob is meant to be *out-of-fold* (evaluate_predictor_cv's
    "oof_proba"), never an in-sample predict_proba(): a model scored on the
    rows it was fit on looks artificially well-calibrated.

    Two design choices, both because this project's calibration sets (Task
    A: ~60 rows, Task B: ~600) are one to two orders of magnitude smaller
    than the vision-benchmark scale (CIFAR/ImageNet) most ECE literature
    assumes:
      - Bin count defaults to round(n^(1/3)) rather than the field's common
        10-15, which is systematically biased at this project's scale
        (Roelofs et al., 2020/2022, AISTATS, "Mitigating Bias in Calibration
        Error Estimation"). Sun et al. (2023, NeurIPS, "Minimum-Risk
        Recalibration of Classifiers") derive n^(1/3) as the risk-minimizing
        bin count under uniform-mass binning -- ~4 bins at Task A's n=60,
        ~8 at Task B's n=600.
      - strategy="quantile" (equal-mass bins) is the adaptive-binning
        default Nixon et al. (2019, "Measuring Calibration in Deep
        Learning") recommend over equal-width ("uniform") for stability
        when predicted probabilities cluster unevenly, which they do here
        (this project's ~20-25% base rate skews scores toward the low end).

    ECE is the standard sample-weighted definition (each bin contributes in
    proportion to how many rows fall in it, not a flat per-bin average) --
    unweighted variants understate error concentrated in a few frequently-hit
    bins. Quantile bin edges can collapse (duplicate edges) when y_prob has
    fewer distinct values than n_bins; the actual bin count used
    ("n_bins_actual") is reported separately from the request ("n_bins")
    for exactly that case.

    Args:
        y_true: Binary ground-truth labels (0/1), shape (n_rows,).
        y_prob: Predicted probability of the positive class, shape (n_rows,).
            Use out-of-fold predictions, not an in-sample fit/score.
        n_bins: Requested bin count. None (default) uses
            max(2, round(len(y_true) ** (1/3))).
        strategy: "quantile" (equal-mass, default) or "uniform" (equal-width).
    Returns:
        Dict with "n_bins" (requested), "n_bins_actual", "strategy",
        "bin_edges", "bin_count" (rows per bin), "bin_mean_predicted",
        "bin_fraction_positive" (parallel lists, one entry per non-empty
        bin, reliability-diagram-ready), "ece", "mce" (max calibration
        error, the single worst bin's |gap|), "brier_score", "n_rows".
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    finite = np.isfinite(y_prob)
    if not finite.all():
        # oof_proba rows with no assigned fold (e.g. n_splits=1) surface as
        # NaN; drop rather than silently corrupt the mean.
        y_true, y_prob = y_true[finite], y_prob[finite]
    n = len(y_true)
    if n == 0:
        raise ValueError("compute_calibration requires at least one finite row")

    requested_bins = max(2, round(n ** (1 / 3))) if n_bins is None else max(2, n_bins)
    if strategy == "quantile":
        edges = np.unique(
            np.quantile(y_prob, np.linspace(0.0, 1.0, requested_bins + 1))
        )
        if len(edges) < 2:  # every y_prob identical
            edges = np.array([y_prob[0] - 1e-9, y_prob[0] + 1e-9])
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, requested_bins + 1)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}; use 'quantile' or 'uniform'")
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf  # guarantee every row lands in a bin

    bin_idx = np.searchsorted(edges, y_prob, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

    bin_count, bin_mean_predicted, bin_fraction_positive = [], [], []
    ece, mce = 0.0, 0.0
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        frac_pos = float(y_true[mask].mean())
        gap = abs(frac_pos - mean_pred)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bin_count.append(count)
        bin_mean_predicted.append(mean_pred)
        bin_fraction_positive.append(frac_pos)

    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "n_bins": requested_bins,
        "n_bins_actual": len(bin_count),
        "strategy": strategy,
        "bin_edges": [float(e) for e in edges],
        "bin_count": bin_count,
        "bin_mean_predicted": bin_mean_predicted,
        "bin_fraction_positive": bin_fraction_positive,
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier,
        "n_rows": n,
    }


def fit_beta_calibrator(y_true, y_prob):
    """Fit a beta-calibration map on held-out (probability, label) pairs.

    Beta calibration (Kull, Silva Filho & Flach, 2017, AISTATS): logistic
    regression on the features [ln p, -ln(1-p)] — a richer parametric
    family than Platt scaling that (a) CONTAINS the identity map, so an
    already-calibrated input is left (near-)unchanged, where Platt's
    sigmoid family cannot represent identity and can actively uncalibrate
    a calibrated model, and (b) is far less overfitting-prone than
    isotonic regression at this project's per-fold calibration-set sizes.
    Platt is deliberately not offered: Manokhin et al. (2026, large-scale
    benchmark) and Spiess et al. (ICSE 2025, on code) independently report
    Platt degrading proper scores. Implementation is ~5 lines of existing
    sklearn — no new dependency (CLAUDE.md Section 4).

    Fit on out-of-fold predictions only (e.g. evaluate_predictor_cv's
    "oof_proba" computed within an outer-fold's training side) — fitting a
    calibrator on the same predictions it will be evaluated on defeats the
    measurement, same leakage rule as compute_calibration's input.

    Args:
        y_true: Binary labels (0/1) for the calibration set.
        y_prob: Predicted positive-class probabilities for the same rows.
    Returns:
        A callable mapping an array of probabilities to calibrated
        probabilities (vectorized, clips inputs away from {0, 1}).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    finite = np.isfinite(y_prob)
    y_true, y_prob = y_true[finite], y_prob[finite]
    if len(np.unique(y_true)) < 2:
        # Degenerate calibration set: nothing to fit; identity is the only
        # honest map.
        return lambda p: np.asarray(p, dtype=float)

    eps = 1e-6

    def _features(p: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
        return np.column_stack([np.log(clipped), -np.log(1.0 - clipped)])

    model = LogisticRegression(max_iter=1000)
    model.fit(_features(y_prob), y_true)

    def calibrate(p):
        return model.predict_proba(_features(p))[:, 1]

    return calibrate

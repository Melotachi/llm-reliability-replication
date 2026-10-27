> **Note for the replication package:** this audit was made against an earlier draft, so its table numbers and section references differ from the submitted paper. Match tables by their LaTeX label (e.g. `tab:auc`); `README.md` §4 lists the current numbering. References to `CLAUDE.md`, `AUTHOR_NOTES.md` or the IST package zip point to internal project files not included here.

# Manuscript number audit — IST submission package

> **Note (2026-10-26):** section, paragraph and line locations below refer to the pre-2026-10-26 text. The concision round rewrote the prose but changed no number and no table body (script-checked), so every audited value still holds; only its location may differ.

**Date:** 2026-09-27
**Audited artifact:** `IST_Paper_Package.zip` → `main_joint.tex` (7 tables, 3 figures,
16 pages) and `supplementary_joint.tex` (8 tables, 5 pages), the integrated manuscript
prepared from the weekly reports.
**Purpose:** close item 1 of the package's `AUTHOR_NOTES.md`, which states that its own
arithmetic check "ham deney tekrarı değildir" (is not an experimental re-verification).
Every value below was re-read from the stored JSON in `results/`. No experiment was re-run
and no manuscript number was changed.

**Result: every table cell and every headline prose number checked resolves to a stored
result file. No discrepancy found.** Two documentation gaps are listed in §9.

---

## 1. Table 3 (`tab:auc`) — sample-level ROC-AUC, corrected representation

Each column comes from a **different** predictor run, and the four files are **not** a
single glob: three are `predictor_*_20260804_*`, the CG/HE164 column is
`predictor_humaneval_20260805_000941.json`. Anyone reproducing this will hit that.

| Column | Source file | `source_run_id` | `config.representation` |
|---|---|---|---|
| CG/HE164 | `results/predictor_humaneval_20260805_000941.json` | `humaneval_uncertainty_merged164_20260711` | `corrected` |
| DS/HE60 | `results/predictor_humaneval_20260804_232950.json` | `humaneval_deepseek60_merged_20260712` | `corrected` |
| CG/MBPP-P | `results/predictor_mbpp_20260804_232300.json` | `mbpp_uncertainty_20260712_024909` (CodeGen-350M) | `corrected` |
| DS/MBPP-P | `results/predictor_mbpp_20260804_233512.json` | `mbpp_uncertainty_20260712_085004` (DeepSeek-1.3B) | `corrected` |

Read at `configurations[] where model_type == "logistic" and feature_set == <arm>` →
`summary.roc_auc_mean`.

| Manuscript row | `feature_set` | CG/HE164 | DS/HE60 | CG/MBPP-P | DS/MBPP-P |
|---|---|---|---|---|---|
| Crash rate only | `crash_rate_self_only` | 0.645 → **0.65** | 0.592 → **0.59** | 0.734 → **0.73** | 0.752 → **0.75** |
| Text families | `self_consistency_core` | 0.857 → **0.86** | 0.849 → **0.85** | 0.745 → **0.75** | 0.699 → **0.70** |
| Execution behavior | `execution_self_only` | 0.902 → **0.90** | 0.916 → **0.92** | 0.833 → **0.83** | 0.876 → **0.88** |
| Text + execution | `self_consistency_plus_execution` | 0.932 → **0.93** | 0.925 → **0.92** | 0.877 → **0.88** | 0.887 → **0.89** |

**16/16 PASS.** Note the caption's "corrected-representation" is accurate: the `full`
representation gives 0.93 / 0.93 / 0.86 / 0.89 for the last row, so the table must not be
cross-checked against the `full` numbers recorded in CLAUDE.md §13 Week 6(e).

## 2. Table 4 (`tab:selection`) — best-of-N selection accuracy

| Column | Source file | `source_run_id` |
|---|---|---|
| DS/HE60 | `results/selection_humaneval_20260714_105452.json` | `humaneval_deepseek60_merged_20260712` |
| CG/HE164 | `results/selection_humaneval_20260714_111521.json` | `humaneval_uncertainty_merged164_20260711` |
| CG/MBPP-P | `results/selection_mbpp_20260714_110252.json` | `mbpp_uncertainty_20260712_024909` |
| DS/MBPP-P | `results/selection_mbpp_20260714_110236.json` | `mbpp_uncertainty_20260712_085004` |

Read at `expected_random` and `selection_accuracy.<key>.overall`.

| Manuscript row | JSON key | DS/HE60 | CG/HE164 | CG/MBPP-P | DS/MBPP-P |
|---|---|---|---|---|---|
| Random, expected | `expected_random` | 0.495 | 0.0915 → 0.091 | 0.1533 → 0.153 | 0.2617 → 0.262 |
| Text classifier | `classifier__self_consistency_core` | 0.5667 → 0.567 | 0.0976 → 0.098 | 0.250 | 0.300 |
| Text + execution classifier | `classifier__self_consistency_plus_execution` | 0.650 | 0.1220 → 0.122 | 0.250 | 0.3667 → 0.367 |
| Execution consensus | `execution_heuristic` | 0.700 | 0.1402 → 0.140 | 0.2333 → 0.233 | 0.3667 → 0.367 |
| Oracle | `oracle` | 0.7833 → 0.783 | 0.2195 → 0.220 | 0.3833 → 0.383 | 0.550 |

**20/20 PASS.**

## 3. Table 5 (`tab:context`) — prompt ablation, base labels

Six pool files, read at `summary.{successful_repairs, repair_success_rate, repair_model_calls}`.

| Cohort | Arm | Source file | Repaired | Rate | Calls |
|---|---|---|---|---|---|
| HumanEval | Task only | `humaneval_week9_arm_resample_sample0_qwen15b.json` | 91/148 | 0.61486 → 61.5 | 444 |
| HumanEval | Task + code | `humaneval_week9_arm_code_only_sample0_qwen15b.json` | 65/148 | 0.43919 → 43.9 | 444 |
| HumanEval | Task + code + error | `humaneval_week9_arm_error_nostop_sample0_qwen15b.json` | 54/148 | 0.36486 → 36.5 | 444 |
| MBPP | Task only | `mbpp_week9_arm_resample_sample0_qwen15b.json` | 27/54 | 0.500 → 50.0 | 162 |
| MBPP | Task + code | `mbpp_week9_arm_code_only_sample0_qwen15b.json` | 14/54 | 0.25926 → 25.9 | 162 |
| MBPP | Task + code + error | `mbpp_week9_arm_error_nostop_sample0_qwen15b.json` | 13/54 | 0.24074 → 24.1 | 162 |

**18/18 PASS.** The caption's "same number of replacement calls" is literally true in the
stored pools: all three arms ran `stop_on_pass=False`, so calls are fixed at 3 × 148 = 444
and 3 × 54 = 162.

## 4. Table 6 (`tab:policies`) — joint MBPP allocation

Source: `results/mbpp_week7_smart_repair_comparison.json`, `conditions.<policy>`.
Cohort block confirms `candidates 60 / initially_passing 6 / initially_failing 54`, and
`config.smart_gate_count = 45`, `config.smart_coverage = 0.75`, `config.seed = 42`.

| Policy | `final_passed` | `added_passes` | `attempted_repairs` | `repair_model_calls` | `model_calls_per_added_pass` |
|---|---|---|---|---|---|
| No repair (`baseline`) | 6 | 0 | 0 | 0 | `None` → "--" |
| Repair-All (`repair_all`) | 16 | 10 | 54 | 147 | 14.7 → 14.70 |
| Smart@75 (`smart_repair_at_0.75`) | 13 | 7 | 44 | 120 | 17.1429 → 17.14 |
| Random@75 (`random_repair_at_0.75`) | 15 | 9 | 40 | 106 | 11.7778 → 11.78 |

**20/20 PASS.**

## 5. Table 7 (`tab:targets`) — one score, two targets

Sources: `results/analysis_week8_gate_target_20260829_211126.json` (MBPP) and
`results/analysis_week8_gate_target_humaneval_20260830_100907.json` (HumanEval), at
`targets[] where target == <name>`.

| Cohort | Target | `n` / `n_positive` | `roc_auc` | `bootstrap_95` |
|---|---|---|---|---|
| MBPP | `initial_passed` | 60 / 6 | 0.94444 → 0.944 | [0.80364, 1.0] → [0.804, 1.000] |
| HumanEval | `initial_passed` | 164 / 16 | 0.94679 → 0.947 | [0.88218, 0.99338] → [0.882, 0.993] |
| MBPP | `repair_success` | 54 / 10 | 0.51818 → 0.518 | [0.29796, 0.74324] → [0.298, 0.743] |
| HumanEval | `repair_success` | 148 / 54 | 0.70646 → 0.706 | [0.616, 0.793] |

**16/16 PASS.** `permutation_p` values (1e-4 and 0.8688) also match the prose.

## 6. Held-out MBPP repair — §5.2.1 prose and supplementary Table S5

Source: same file as §4, block `week6_stop_on_pass_counterfactual`.

| Manuscript claim | JSON | Status |
|---|---|---|
| repairs 12 of 50 held-out failures (24.0%) | `completed_failures 50`, `successful_repairs 12` | PASS |
| generates 150 candidates | `actual_model_calls 150` | PASS |
| 136 unique | 150 − 14 duplicates (supp. Table S5) | PASS |
| eight at first, two at second, two at third | `first_passing_candidate_indices {0:8, 1:2, 2:2}` | PASS |
| stop-on-pass replay 132 calls, 11.0 per success | `stop_on_pass_model_calls 132`, `..._calls_per_success 11.0` | PASS |
| ranking oracle gap zero | `repair_all_success_capture_rate 1.0` | PASS |

## 7. Paired context comparisons — §5.2.2 prose and supplementary Table S6

| Label | Source file | `left_only` vs `right_only` | `mcnemar_mid_p_bh` | Manuscript |
|---|---|---|---|---|
| base, none vs error | `analysis_week9_context_mode_ladder_humaneval_base_20260830_195949.json` | 49 vs 12 | 3.6431e-06 | "49 versus 12 … 3.6 × 10⁻⁶" PASS |
| plus, none vs error | `analysis_week9_context_mode_ladder_humaneval_plus_20260830_195959.json` | 47 vs 14 | 5.2286e-05 | "47 versus 14 … 5.2 × 10⁻⁵" PASS |
| base, code vs error | base file above | 16 vs 5 | 0.016901 | "also significant" PASS |
| plus, code vs error | plus file above | 16 vs 3 | 0.003865 | PASS |

## 8. BigCodeBench ladder — §5.1.3 prose and supplementary Table S4

Five per-rung analyses, `results/analysis_week11_bcb_<run>_bigcodebench_*.json`, at
`correlations.<metric>.{spearman_rho, spearman_p_fdr}`; pass fractions recomputed from each
`source_results_file`'s `per_problem`. Stars = `spearman_p_fdr < 0.05`.

| | CG-350M | Qwen-0.5B | Qwen-1.5B | Qwen-3B | Qwen-7B |
|---|---|---|---|---|---|
| Mean pass fraction | .0450 | .1183 | .2800 | .3433 | .4067 |
| Problems with a pass | 11 | 20 | 38 | 36 | 41 |
| Parse-failure fraction | .0700 | .0150 | .0067 | .0017 | .0000 |
| AST-sequence ρ | −.190 | −.266* | −.438* | −.162 | −.153 |
| Token-Jaccard ρ | −.207 | −.272* | −.334* | −.316* | −.309* |
| Token-cosine ρ | −.229 | −.379* | −.426* | −.349* | −.302* |
| Lexical-diversity ρ | .061 | −.317* | −.270* | −.103 | −.106 |

**All 35 values and all 20 significance stars PASS.** Confirms the §5.1.3 prose "significant
after the reported correction at only two of five model sizes" for AST-sequence and "four"
for token Jaccard, and "pass fractions increase from 0.0450 to 0.4067".

## 9. Two documentation gaps — text to add, not numbers to change

**(a) `n = 147` under strengthened labels** (`AUTHOR_NOTES.md` item 3, previously open).
Resolved from the data, not inferred:
`results/analysis_week9_evalplus_relabel_humaneval_20260830_195445.json` records
`config.excluded_tasks = ["HumanEval/32"]` and `initial_pool.skipped_task_ids =
["HumanEval/32"]`, with `cells[].n_coordinates_total = 148` and `n_coordinates_scored = 147`.
**`HumanEval/32` has no usable EvalPlus entry and is excluded from both the base and the
strengthened label, so every strengthened-label rate runs over 147, not 148.** The manuscript
should state this once where the plus counts first appear. Do NOT convert this into a call
count: the recorded figure is 444 calls per arm (3 × 148) under both labels, because the calls were
spent before relabeling. Only the scored denominator changes, 148 → 147. An earlier version of this
audit derived “441 calls” by arithmetic; no results JSON records it, and it contradicts
`tab:context`, so it was removed from the manuscript on 2026-09-27.

**(b) The BigCodeBench table's representation is not named.** Its five source runs carry
`config.representation = "full"`, whereas Table 3 is `corrected`. Per the project's own rule
(CLAUDE.md Week 10(b)), any cross-cell comparison of `parse_failure_rate` must name its
representation. One clause in the Table S4 caption fixes it.

## 10. What this audit does not cover

- Figure 2 and Figure 3 are redrawn from `figure_data.json` aggregates, not from `results/`
  directly; the underlying counts are the §3 and §4 rows above, which pass.
- Table 1 (`tab:position`) and Table 2 (`tab:cohorts`) carry no experimental estimates; the
  cohort sizes in Table 2 (60/54, 164/148, 50, 444/162, 60 × 5 rungs) match §3–§6.
- Confidence intervals other than those listed, secondary prose statistics in §5.1.2,
  §5.3.3 and §6, and every supplementary table beyond S4/S5/S6 were not re-derived.
- Reproduction metadata (public repository URL, immutable commit, archival DOI) is still
  absent, exactly as the manuscript's own Data availability section states.

---

# Addendum — numbers added in the 2026-10-08 round

Added for the supervisor's fifth comment batch (table highlighting, baseline
comparison, extra per-RQ evidence, answer boxes, statistical tests). As above,
every value is re-read from stored JSON in `results/`; nothing was re-run.
Table numbers below are the ones in the 26-page version of `main_joint.tex`.

## A1. Table 4 (`tab:problemauc`) — problem-level ROC-AUC

Same four predictor files as Table 3 (§1), same `corrected` representation, same
five-fold grouped logistic model. Read at
`configurations[] where task == "problem_level" and model_type == "logistic" and
feature_set == <arm>` → `summary.roc_auc_mean` and `summary.roc_auc_std`.

| Manuscript row | `feature_set` | CG/HE164 | DS/HE60 | CG/MBPP | DS/MBPP |
|---|---|---|---|---|---|
| Completion length only | `length_only` | .854 ±.076 | .904 ±.072 | .713 ±.108 | .446 ±.187 |
| Crash rate only | `crash_rate_only` | .786 ±.066 | .723 ±.052 | .791 ±.140 | .729 ±.078 |
| Text families | `no_codebert` | .828 ±.081 | .926 ±.049 | .728 ±.081 | .583 ±.178 |
| Execution behavior | `execution_only` | .788 ±.101 | .939 ±.051 | .627 ±.261 | .649 ±.106 |
| Text + execution | `no_codebert_plus_execution` | .851 ±.044 | .969 ±.029 | .683 ±.072 | .647 ±.177 |

**20/20 PASS.** The "free baseline meets or exceeds the fusion arm in three of
four cells" claim is arithmetic on this table: .854 ≥ .851, .791 ≥ .683,
.729 ≥ .647; DS/HE60 is the exception (.904 and .723 against .969).
The `Chance` row is 0.50 by definition, not a measured value.

## A2. Table 5 (`tab:selection`) — the derived headroom row

Arithmetic on the rows above it, `(best − random) / (oracle − random)`:

| Cell | best | random | oracle | headroom captured |
|---|---|---|---|---|
| CG/HE164 | 0.140 | 0.091 | 0.220 | 0.049/0.129 = 0.380 → **0.38** |
| DS/HE60 | 0.700 | 0.495 | 0.783 | 0.205/0.288 = 0.712 → **0.71** |
| CG/MBPP | 0.250 | 0.153 | 0.383 | 0.097/0.230 = 0.422 → **0.42** |
| DS/MBPP | 0.367 | 0.262 | 0.550 | 0.105/0.288 = 0.365 → **0.37** |

No new source file. The row is labelled "(derived)" in the table.

## A3. Table 6 (`tab:context`) — regeneration baseline, intervals, unit price, tests

New analysis script `scripts/compare_regeneration_baseline.py`; output
`results/analysis_regeneration_baseline_20261008_234528.json`. It asserts
`config.selection_sha256` on every pool against the cohort's expected value
(`9332324b…` HumanEval, `85c09d30…` MBPP) and asserts that all three arms carry
an identical coordinate list before anything is computed.

**Both cohorts use the Week 9 pools for all three arms**, which is what makes the
call accounting match the manuscript's fixed-budget convention (3 candidates per
failure, no early stop → 444 and 162 calls). The Week 7/8 pools are deliberately
not used: they stopped on the first passing candidate, and the MBPP Week 7 pool
is a different pool (10/54) from the manuscript's row (13/54).

| Row | Source | Value |
|---|---|---|
| Resample baseline, HumanEval | `humaneval_uncertainty_merged164_20260711.json`, samples 1–3 | 12/148 = 8.1% [4.7, 13.6], 444 generator calls, 37.0 calls/success |
| Resample baseline, MBPP | `mbpp_uncertainty_20260712_024909.json`, samples 1–3 | 11/54 = 20.4% [11.8, 32.9], 162 generator calls, 14.73 calls/success |
| Calls/success, HumanEval arms | pools' `summary.model_calls_per_success` | 4.88 / 6.83 / 8.22 |
| Calls/success, MBPP arms | pools' `summary.model_calls_per_success` | 6.00 / 11.57 / 12.46 |
| Wilson 95%, all arms | pools' `summary.repair_success_wilson_95` | as printed |
| Paired mid-*p* vs baseline, HumanEval | `arms[].paired_vs_resample` | 81–2 → 1.1e-21; 59–6 → 4.1e-12; 47–5 → 7.1e-10 (BH within cohort) |
| Paired mid-*p* vs baseline, MBPP | same | 17–1 → 2.3e-4; 9–6 → 0.58; 7–5 → 0.58 (BH) |

The resample k is fixed at 3 and only 3: it is the only value that matches the
arms' three replacement candidates per failure. The k = 3 rows agree with the
independently computed `resample_control` of
`analysis_week12_arm_allocation_20260907_212130.json` (HumanEval 12, MBPP 11).

**Caveat that must stay in the caption and the prose:** baseline calls go to
CodeGen-350M, arm calls go to Qwen2.5-Coder-1.5B-Instruct. The arms are matched
on coordinates and on call count, not on model.

## A4. Table 7 (`tab:category`) — repairs by failure category

`results/analysis_week9_context_mode_ladder_humaneval_base_20260830_195949.json`
and `…_mbpp_base_20260831_105751.json`, read at
`category_breakdown.by_arm[<arm>][<category>].successful_repairs` and
`.failures`.

| Cohort | Category | Failures | none | code | error |
|---|---|---|---|---|---|
| HumanEval | assertion_error | 116 | 73 | 57 | 46 |
| | name_error | 15 | 9 | 3 | 2 |
| | runtime_error | 8 | 4 | 4 | 5 |
| | syntax_error | 5 | 3 | 0 | 0 |
| | type_error | 4 | 2 | 1 | 1 |
| MBPP | assertion_error | 33 | 18 | 8 | 10 |
| | name_error | 7 | 4 | 4 | 2 |
| | runtime_error | 4 | 2 | 1 | 0 |
| | syntax_error | 3 | 1 | 0 | 0 |
| | type_error | 7 | 2 | 1 | 1 |

**30/30 PASS.** Category totals reconcile with the arm totals: HumanEval
73+9+4+3+2 = 91, 57+3+4+0+1 = 65, 46+2+5+0+1 = 54; MBPP 18+4+2+1+2 = 27,
8+4+1+0+1 = 14, 10+2+0+0+1 = 13. Failure counts sum to 148 and 54. The prose
percentages 116/148 = 78.4% and 33/54 = 61.1% are arithmetic on this table.

## A5. Table 10 (`tab:armalloc`) — allocation signal per prompt arm

`results/analysis_week12_arm_allocation_20260907_212130.json`, read at
`cohorts[humaneval].arms[<arm>].targets.repair_success`.

| Arm | ROC-AUC | permutation *p* | bootstrap 95% | positives |
|---|---|---|---|---|
| none | 0.5417 → **0.542** | 0.388 → **0.39** | [0.4435, 0.6358] | 91 |
| code_only | 0.6482 → **0.648** | 0.00160 → **0.0016** | [0.5561, 0.7370] | 65 |
| error | 0.7065 → **0.706** | 9.999e-5 → **1.0e-4** | [0.6164, 0.7932] | 54 |

10,000 permutations and 10,000 bootstrap resamples, as recorded in the file.
The `error` row equals Table 9's HumanEval repair-success row, as the caption says.

**MBPP, reported in prose only:** `none` 0.4815 → **0.481**, bootstrap
[0.3224, 0.6386], 27 positives; `code_only` 0.675, *p* = 0.0544 → **0.054**, 14
positives. **The MBPP `error` cell is deliberately not reported**: the only
stored allocation pool for it is `mbpp_week7_repair_all_sample0_qwen15b.json`
(10/54), which is not the pool behind Table 6's MBPP `error` row (13/54), and it
carries ten positives. Printing it would put two different call counts and two
different success counts in the paper for one arm.

## A6. The one-sample *t*-tests on the random allocation replays

Computed in the same new script from the reported mean, standard deviation and
*n* of the 200 coverage-matched replays already in Table S7, against the
uncertainty gate's single observed value. Cross-checked against
`scipy.stats.t.sf` to full printed precision.

| Cohort | mean ± sd (n = 200) | gate | *t* | df | two-sided *p* |
|---|---|---|---|---|---|
| MBPP | 7.56 ± 1.19 | 7 | 6.655 → **6.66** | 199 | 2.69e-10 → **2.7e-10** |
| HumanEval | 40.5 ± 2.7 | 36 | 23.570 → **23.57** | 199 | 1.67e-59 → **1.7e-59** |

This is a statement about the replay distribution over one frozen pool, as the
manuscript says in both §4.2 and §5.3.1. It is not a population test, and the
82.5% / 97.5% dominance percentages remain the primary summary.

## A7. What this addendum does not cover

- The shading in Tables 3–7 and 10 encodes only the ordering of values already
  audited above; no shaded cell carries a number that is not in a row checked here.
- The answer boxes after each research question restate numbers from §5 and the
  abstract. Every figure in them was matched by string search against the body
  text; none is newly derived.

## A8. Two call-accounting conventions coexist in Section 5, deliberately

Table 6 and Table 10 use the **fixed-budget** convention: three candidates per
failure, no early stop, so the HumanEval arms cost 444 calls and the MBPP arms
162. Section 5.3.3 and supplementary Table S10 use the **stop-on-pass**
convention, under which the same stored candidate set for the code-plus-error
arm costs 368 calls, because a coordinate repaired on its first candidate does
not pay for the other two. Candidate seeds are index-deterministic, so both
counts are exact for one candidate set and neither is an estimate. Both
conventions now say so where they appear (the `tab:armalloc` caption and the new
lead sentence of §5.3.3), because Table 10 and §5.3.3 are adjacent and a reader
who did not know would read 444 against 368 as a contradiction. Cross-check:
$91 \times 4.88 = 444$ and $54 \times 8.22 = 444$, which is the fixed-budget
arithmetic; the 368 figure is the stop-on-pass replay recorded in
`analysis_week12_arm_allocation_20260907_212130.json` under
`arms.error.call_renormalization`.

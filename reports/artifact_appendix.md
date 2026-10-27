> **Note for the replication package:** these notes were written during the project and use its internal objective/week numbering and file paths (e.g. `reports/overleaf/`, weekly reports) that are not part of this package. For the mapping from the published paper's tables and figures to scripts and result files, see `README.md` §4.

# Artifact appendix

Companion to `reports/overleaf/main_joint.tex`. Everything a reviewer needs to
re-derive the paper's numbers, and an honest statement of what cannot be
re-derived without a GPU.

- **Repository state this appendix describes:** the commit that adds this file (2026-09-01),
  branch `feature/student-a-capability-7b-rung`.
- **Archive:** not yet minted. Before submission, upload a tag of this
  repository (including `results/`, 39 MB) to Zenodo and replace the DOI
  placeholder in `main_joint.tex` §Reproducibility. The repository is
  self-contained: `results/` is version-controlled (183 JSON files), so no
  external download is required for any re-analysis.

---

## 1. What is in the artifact

| Path | Contents |
|---|---|
| `src/` | The six library modules (`generation`, `uncertainty`, `selective`, `repair`, `evaluation`, `utils`). |
| `scripts/` | One CLI per experiment or analysis. Every paper number is produced by one of these. |
| `results/` | 183 persisted JSON runs — per-sample generations, per-candidate repair traces, and every analysis output. **This is the artifact's substance:** every re-analysis in the paper reads these, not a model. |
| `reports/` | The weekly reports, the novelty verification, the claim/scope table, the two pre-registrations, and `figures/`. |
| `reports/overleaf/` | `main.tex` (Objective 1 weekly log), `main_student_b.tex` (Objective 2 weekly log), `main_joint.tex` (the paper). |
| `tests/` | 29 files, 466 tests, all passing at that commit (`python -m pytest tests/ -q`, ~66 s on CPU). |

## 2. Environment

Pinned in `requirements.txt`; the versions the reported runs were produced under:

```
python 3.13.1          # CLAUDE.md floor is 3.10; nothing below 3.10 is supported
datasets==5.0.0        transformers==4.50.3     torch==2.6.0
accelerate==1.12.0     numpy==2.1.3             scikit-learn==1.6.1
scipy==1.15.2          matplotlib==3.10.1       seaborn==0.13.2
zss==1.2.0             tree-sitter==0.26.0      tree-sitter-python==0.25.0
pytest==9.1.1          black==26.5.1            isort==8.0.1
```

Hardware actually used, and it matters — see §5:

| Tier | Hardware | Used for |
|---|---|---|
| Local | macOS (Darwin 25.5), CPU | every re-analysis, every test, the early Week 1–6 generations |
| Hosted GPU | Kaggle, **Tesla P100 (sm_60)**, 16 GB | the capability ladder, the ≥7B rung, all Qwen-1.5B repair runs |

Two hosted-GPU traps are encoded in the driver scripts rather than left to the
reader. (i) An API-pushed Kaggle kernel gets a P100, not the UI's T4, and the
image's `torch 2.10+cu128` ships no `sm_60` kernels — the first `generate()`
dies with `CUDA error: no kernel image is available for execution on the
device`, reported asynchronously and easily mistaken for a hang. The drivers
pin `torch 2.6.0+cu124` when the capability is `(6, 0)`, after removing
`torchvision`/`torchaudio`. (ii) An API-pushed kernel mounts its dataset at
`/kaggle/input/datasets/<owner>/<slug>/<version>/`, not `/kaggle/input/<slug>/`;
the drivers search from `/kaggle/input`.

## 3. Reproduction tiers

**Tier 1 — every analysis in the paper, CPU only, no model download.**
This is the whole of Objectives 1 and 3's analysis and all of Objective 2's
post-hoc analysis: each script reads stored per-sample records. Minutes, not
hours. Representative commands:

```bash
# Objective 1 — the seven-cell prompt-confound re-analysis
python scripts/compare_representations.py --seed 42

# Objective 1 — the capability ladder against the pre-registered predictions
python scripts/analyze_capability_sweep.py --seed 42

# Objective 1 — the EvalPlus relabel of every rung (strict-label control)
python scripts/rescore_evalplus_labels.py --sensitivity-timeout 120

# Objective 1 — split-half reliability and the attenuation ceiling
python scripts/estimate_reliability.py --n-permutations 1000 --seed 42

# Objective 3 — the paired four-policy comparison and the 200-seed control
python scripts/analyze_smart_repair.py \
  --source-results results/mbpp_uncertainty_20260712_024909.json \
  --predictor-results results/predictor_mbpp_20260712_212240.json \
  --repair-results results/mbpp_week7_repair_all_sample0_qwen15b.json \
  --sample-index 0 --smart-coverage 0.75 --seed 42

# Week 10 — routing between the repair arms (negative result)
python scripts/route_context_arms.py --arms \
  results/humaneval_week9_arm_resample_sample0_qwen15b.json \
  results/humaneval_week9_arm_code_only_sample0_qwen15b.json \
  results/humaneval_week9_arm_error_nostop_sample0_qwen15b.json
```

**Tier 2 — regenerating samples. GPU, and hours.** Only needed to re-derive the
stored runs themselves. Wall clock measured on the P100: Qwen2.5-Coder-3B over
60 problems × 10 samples = 1,336 s; Qwen2.5-Coder-7B (offloaded across
`['0', 'cpu']`, which is why it is 9× slower) = 12,105 s; the three Week 9
repair arms = 3 h 48 m total.

```bash
python scripts/run_pipeline.py --mode uncertainty --dataset humaneval \
  --n-problems 60 --n-samples 10 --temperature 0.7 --seed 42 \
  --model Qwen/Qwen2.5-Coder-3B --device cuda --dtype float32
```

**Tier 3 — the test suite.** `python -m pytest tests/ -q`. Tests mock model
loading; the handful of real-encoder tests are marked `slow`.

## 4. Claim → artefact map

One row per paper section, naming the script that produces it and the stored
input it reads. Every output listed is itself in `results/`.

| Paper section | Script | Key input(s) | Output |
|---|---|---|---|
| RQ1(a) base correlations | `analyze_uncertainty.py` | `humaneval_uncertainty_20260710_014920.json` | `analysis_week4_humaneval_*.json` |
| RQ1(b) prompt confound, 7 cells | `compare_representations.py` | the seven cell runs | `analysis_week7_representations_20260804_221056.json` |
| RQ1(c) length baseline + ceiling | `analyze_uncertainty.py`, `estimate_reliability.py` | same seven cells | confound table; reliability JSON |
| RQ1(d) transfer matrix | `analyze_uncertainty.py` | `mbpp_uncertainty_*`, `humaneval_deepseek60_merged_*`, `humaneval_t03_60_merged_*` | `analysis_week6_*.json` |
| RQ2 predictor + calibration | `train_predictor.py`, `calibrate_predictor.py` | `humaneval_uncertainty_merged164_20260711.json` | `predictor_humaneval_20260712_215312.json`, `calibration_humaneval_*.json` |
| RQ3 ablation | `train_predictor.py` | the four cells | `predictor_{humaneval,mbpp}_*_20260712_*.json` |
| RQ3(b) decision metrics | `evaluate_selective_prediction.py` | stored OOF probabilities | `analysis_week7_selective_20260804_222527.json` |
| RQ3(c) best-of-N selection | `evaluate_selection.py` | the four cells' stored samples | `selection_*_20260714_*.json` |
| RQ3(d) consensus deciles | `analyze_consensus_deciles.py` | `humaneval_uncertainty_merged164_20260711.json` | consensus JSON |
| §crash control | `diagnose_crash_control.py` | the four cells | `analysis_week7_crashcontrol_20260805_*.json` |
| RQ4 capability ladder (7 rungs) | `analyze_capability_sweep.py` | `humaneval_uncertainty_20260827_*`, `_20260831_212704`, `_20260901_004849` | `analysis_week7_capability_20260901_115610.json` |
| RQ4 EvalPlus relabel | `rescore_evalplus_labels.py` | the same seven rungs | `analysis_week7_evalplus_relabel_20260901_113520.json` |
| RQ4 N-sweep | `preregister_n_sweep.py`, `analyze_n_sweep.py` | the two 50-sample pools | `prereg_n_sweep.json`, N-sweep JSON |
| RQ5–RQ6 repair | `run_iterative_repair_experiment.py`, `analyze_repair_robustness.py` | `mbpp_week6_failure_pool_seed42.json` | `mbpp_week6_*`, robustness JSON |
| RQ7(b) context ladder (HumanEval) | `analyze_context_mode_ladder.py` | `humaneval_week9_arm_{resample,code_only,error_nostop}_*.json` | `analysis_week9_context_mode_ladder_humaneval_{base,plus}_*.json` |
| RQ7(b) context ladder (MBPP) | same | `mbpp_week9_arm_*.json` | `analysis_week9_context_mode_ladder_mbpp_{base,plus}_*.json` |
| Objective 3, cohort 1 | `analyze_smart_repair.py`, `sweep_random_gate.py`, `diagnose_gate_target.py` | `mbpp_week7_repair_all_sample0_qwen15b.json` | `mbpp_week7_smart_repair_comparison.json`, `analysis_week8_*.json` |
| Objective 3, cohort 2 | `diagnose_gate_direction.py` | `humaneval_week8_repair_all_sample0_qwen15b.json` | gate-direction JSON |
| Week 10 arm routing | `route_context_arms.py` | the three Week 9 arms + `predictor_humaneval_20260712_215312.json` | `analysis_week10_router_{base,plus}_*.json` |

## 5. Determinism, and the three places it is not free

1. **Seeding.** Every entry point calls `set_all_seeds(seed)` (`random`, `numpy`,
   `torch`, CUDA), default 42. Repair candidate seeds are *index-deterministic*,
   which is why a `--resume`d arm is bit-identical to an uninterrupted one, and
   why `--stop-on-pass` changes the call count without changing which candidates
   exist.
2. **Device and dtype are part of the experiment, not of the environment.**
   Re-generating three ladder rungs on CUDA that had been produced on CPU moved
   `pass@1` 0.230 → 0.2167 and 0.495 → 0.4467 and changed a frozen fit's
   anchors. Runs therefore persist `device_requested`/`device_resolved` and
   `dtype_requested`/`dtype_resolved`, and `analyze_capability_sweep.py`'s
   `PROTOCOL_KEYS` include both, so a mismatched rung is refused rather than
   silently joined. Any cross-rung comparison must be device- and dtype-matched.
3. **Stored paths were Windows-shaped.** The Week 6/7 traces were generated on
   Windows and carry `results\file.json`; analyses parse them with
   `PureWindowsPath`. Without that, no Week 7 Objective 3 number reproduced on
   macOS or Linux.

Verified cross-platform: the Week 8 second-cohort analyses were produced on the
hosted GPU host and re-run on macOS from the same stored artefacts, reproducing
36 added passes at 312 calls, ROC-AUC 0.9468 and 0.7065, and 97.5% domination
exactly.

## 6. Data provenance

| Dataset | Source | Use |
|---|---|---|
| HumanEval | `openai_humaneval` via `datasets` | primary benchmark, problems 0–163 |
| MBPP | `mbpp` via `datasets` | second benchmark; the MBPP+-member subset is pinned in `data/mbppplus_member_task_ids.txt` |
| HumanEval+ / MBPP+ | `evalplus/humanevalplus`, `evalplus/mbppplus` | strict relabelling and the EvalPlus probe inputs |

Models are all open-weight and free: `Salesforce/codegen-350M-mono`,
`deepseek-ai/deepseek-coder-1.3b-base`, `Qwen/Qwen2.5-Coder-{0.5B,1.5B,3B,7B}`,
`Qwen/Qwen2.5-Coder-1.5B-Instruct` (the repairer), `microsoft/codebert-base`
(the semantic metric family). No paid API is called anywhere in `src/` or
`scripts/`.

## 7. Safety

Generated code is executed. `run_code()` uses a timeout-bounded subprocess and
is marked `# TODO: SANDBOX`: it is isolated in time but not in filesystem or
network access, which is acceptable for local research use and is **not** a
sandbox. Reviewers running Tier 2 should do so in a container.

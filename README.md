# Replication package: From Uncertainty Estimation to Repair Allocation in Code Generation

Replication package for the paper

> Melih Elibol, Ayşenur Gönül, Sushant Kumar Pandey.
> *From Uncertainty Estimation to Repair Allocation in Code Generation: An Empirical Study.*
> Under submission to Information and Software Technology.

This package has all the code, stored experiment records, task identifiers and
instructions needed to reproduce every number, table and figure in the paper and
its supplementary material. The paper and supplement PDFs are in `paper/`.

The study uses only open-weight models from the HuggingFace Hub and public
benchmarks. No paid API and no closed-weight model is used anywhere.

---

## 1. What is in the package

| Path | Contents |
|---|---|
| `src/` | Library modules: `generation` (model loading, multi-sample generation), `uncertainty` (all uncertainty metrics, predictors, calibration), `selective` (risk-coverage / AUGRC), `evaluation` (execution engine, pass@k, behavior signatures, McNemar mid-p), `repair` (repair prompts and candidate ranking), `utils` (seeding, I/O, dataset loading). |
| `scripts/` | One command-line entry point per experiment or analysis. Every reported number is produced by one of these. |
| `results/` | 240 persisted JSON records: per-sample generations, per-candidate repair traces and every analysis output. **This is the core of the package.** All analyses in the paper read these records and need no model. |
| `data/` | Pinned task-identifier lists (`mbppplus_member_task_ids.txt`, `bigcodebench_cell_task_ids.txt`, `bigcodebench_eligible_task_ids.txt`), the cached BigCodeBench release used for the BigCodeBench cell, and the SQLite fixtures some BigCodeBench tests need (`task_func/`). HumanEval, MBPP, HumanEval+ and MBPP+ are downloaded into the empty `data/*/` folders on first use. |
| `notebooks/` | The Kaggle drivers used for the GPU runs (generation ladder, 7B rung, repair arms, BigCodeBench cell, EvalPlus relabel). |
| `tests/` | Unit tests (pytest). Model loading is mocked, so the suite runs on CPU. |
| `paper/` | `main_joint.pdf`, `supplementary_joint.pdf`, and `make_figures.py` + `figure_data.json`, which regenerate the paper's two vector plots (`paper/figures/`). |
| `reports/artifact_appendix.md` | Longer artifact notes: environment, wall-clock times, determinism, data provenance. |
| `reports/paper_number_audit.md` | A cell-by-cell audit of every table and headline number in the paper against the JSON field it comes from. Use this to check any single number. |

## 2. Requirements

- Python 3.13 (the package was verified on a clean Python 3.13 environment).
- The exact package versions are pinned in `requirements.txt`.
- **Reproducing the paper's analyses (Tier 1 below) needs only a CPU.** About 4 GB of
  disk for the Python environment. No model is downloaded.
- Regenerating the stored samples (Tier 2) needs a CUDA GPU. The runs in the paper
  were made on a 16 GB Tesla P100 (Kaggle). The 7B model does not fit in 16 GB
  and is loaded with `--device-map auto` (CPU offloading).

```bash
git clone https://github.com/Melotachi/llm-reliability-replication.git
cd llm-reliability-replication
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the repository root.

## 3. Reproducing the results

### Tier 0: check the installation (about 30 s)

```bash
python -m pytest tests/ -q -m "not slow"
```

Expected: `612 passed, 1 skipped, 5 deselected`. The `slow` tests load the real
CodeBERT encoder (about 500 MB download).

### Tier 1: every analysis in the paper (CPU only, minutes)

Each script reads the stored records in `results/` and writes a new timestamped
JSON to `results/`. Compare the new file with the stored file named in the comment
(`run_id` and `timestamp` will differ). Figures are written to `reports/figures/`.
The parameters below are the ones recorded in each stored file's `config` block.

Commands marked **[replays code]** re-execute stored generated programs to compute
execution features (see Section 7) and take several minutes per cell on a laptop.
All others are pure re-analysis and finish in seconds.

```bash
# --- RQ1: correctness estimation and selection --------------------------------

# Tables 3-4: sample- and problem-level ROC-AUC, corrected representation. One run
# per cell; stored results in parentheses.                              [replays code]
python scripts/train_predictor.py --results results/humaneval_uncertainty_merged164_20260711.json \
  --no-codebert --representation corrected --probe-source evalplus   # predictor_humaneval_20260805_000941
python scripts/train_predictor.py --results results/humaneval_deepseek60_merged_20260712.json \
  --no-codebert --representation corrected --probe-source evalplus   # predictor_humaneval_20260804_232950
python scripts/train_predictor.py --results results/mbpp_uncertainty_20260712_024909.json \
  --no-codebert --representation corrected --probe-source evalplus   # predictor_mbpp_20260804_232300
python scripts/train_predictor.py --results results/mbpp_uncertainty_20260712_085004.json \
  --no-codebert --representation corrected --probe-source evalplus   # predictor_mbpp_20260804_233512

# Table 5: best-of-N selection, one run per cell.                       [replays code]
python scripts/evaluate_selection.py --results results/humaneval_deepseek60_merged_20260712.json --no-codebert   # selection_humaneval_20260714_105452
python scripts/evaluate_selection.py --results results/humaneval_uncertainty_merged164_20260711.json --no-codebert # selection_humaneval_20260714_111521
python scripts/evaluate_selection.py --results results/mbpp_uncertainty_20260712_024909.json --no-codebert        # selection_mbpp_20260714_110252
python scripts/evaluate_selection.py --results results/mbpp_uncertainty_20260712_085004.json --no-codebert        # selection_mbpp_20260714_110236

# Supplementary Table S1: correlations before/after prompt removal, seven cells   [~2 h, CPU-bound]
python scripts/compare_representations.py --no-codebert --seed 42 \
  --run cg_he60=results/humaneval_uncertainty_20260710_014920.json \
  --run cg_he164=results/humaneval_uncertainty_merged164_20260711.json \
  --run cg_t03_60=results/humaneval_t03_60_merged_20260712.json \
  --run cg_mbpp=results/mbpp_uncertainty_20260711_021511.json \
  --run cg_mbppplus=results/mbpp_uncertainty_20260712_024909.json \
  --run ds_he60=results/humaneval_deepseek60_merged_20260712.json \
  --run ds_mbppplus=results/mbpp_uncertainty_20260712_085004.json   # analysis_week7_representations_20260804_221056

# Supplementary Table S2: calibration with beta recalibration          [replays code, ~20 min each]
python scripts/calibrate_predictor.py --results results/humaneval_uncertainty_20260710_014920.json \
  --recalibrate beta          # calibration_humaneval_20260713_153953 (uses CodeBERT, ~500 MB download)
python scripts/calibrate_predictor.py --results results/humaneval_uncertainty_merged164_20260711.json \
  --recalibrate beta --no-codebert   # calibration_humaneval_20260713_155535

# --- RQ2: repair and failure context -----------------------------------------

# Table 6 and Figure 3: matched-call regeneration baseline
python scripts/compare_regeneration_baseline.py   # analysis_regeneration_baseline_20261008_234528

# Table 7 and Supplementary Table S5: prompt ablation, paired tests, failure categories
python scripts/analyze_context_mode_ladder.py --dataset humaneval --label base \
  --arms results/humaneval_week9_arm_resample_sample0_qwen15b.json \
         results/humaneval_week9_arm_code_only_sample0_qwen15b.json \
         results/humaneval_week9_arm_error_nostop_sample0_qwen15b.json \
  --week8-results results/humaneval_week8_repair_all_sample0_qwen15b.json
                                   # analysis_week9_context_mode_ladder_humaneval_base_20260830_195949
python scripts/analyze_context_mode_ladder.py --dataset mbpp --label base \
  --arms results/mbpp_week9_arm_resample_sample0_qwen15b.json \
         results/mbpp_week9_arm_code_only_sample0_qwen15b.json \
         results/mbpp_week9_arm_error_nostop_sample0_qwen15b.json
                                   # analysis_week9_context_mode_ladder_mbpp_base_20260831_105751
# Strengthened (EvalPlus) labels: same commands with --label plus and the
# *_evalpluslabel_*.json arm files listed in the stored analysis's config.arms.

# --- RQ3: repair allocation --------------------------------------------------

# Table 8 and Supplementary Table S4: joint MBPP allocation, held-out repair accounting
python scripts/analyze_smart_repair.py \
  --source-results results/mbpp_uncertainty_20260712_024909.json \
  --predictor-results results/predictor_mbpp_20260712_212240.json \
  --repair-results results/mbpp_week7_repair_all_sample0_qwen15b.json \
  --sample-index 0 --smart-coverage 0.75 --seed 42   # mbpp_week7_smart_repair_comparison

# Supplementary Table S6: 200 coverage-matched random gates (MBPP, then HumanEval)
python scripts/sweep_random_gate.py               # analysis_week8_random_gate_sweep_20260829_211142
python scripts/sweep_random_gate.py \
  --source-results results/humaneval_uncertainty_merged164_20260711.json \
  --predictor-results results/predictor_humaneval_20260712_215312.json \
  --repair-results results/humaneval_week8_repair_all_sample0_qwen15b.json \
  --prefix analysis_week8_random_gate_sweep_humaneval   # analysis_week8_random_gate_sweep_humaneval_20260830_100905

# Table 9: one score, two targets (MBPP, then HumanEval)
python scripts/diagnose_gate_target.py            # analysis_week8_gate_target_20260829_211126
python scripts/diagnose_gate_target.py \
  --source-results results/humaneval_uncertainty_merged164_20260711.json \
  --predictor-results results/predictor_humaneval_20260712_215312.json \
  --repair-results results/humaneval_week8_repair_all_sample0_qwen15b.json \
  --prefix analysis_week8_gate_target_humaneval   # analysis_week8_gate_target_humaneval_20260830_100907

# Table 10: allocation signal per prompt arm, both cohorts
python scripts/diagnose_arm_allocation.py         # analysis_week12_arm_allocation_20260907_212130

# Figure 4 and Supplementary Table S7: allocation under call caps
python scripts/pilot_gate_direction.py            # analysis_week12_pilot_direction_20260907_222814

# Paper figures from the reported aggregates
cd paper && python make_figures.py && cd ..
```

**What was verified on a clean copy of this package** (fresh Python 3.13
environment from `requirements.txt`, macOS, CPU), comparing every field of the new
JSON with the stored one except `run_id`/`timestamp`:

| Command (stored result) | Outcome | Time |
|---|---|---|
| `train_predictor.py`, CG/MBPP+ cell (`predictor_mbpp_20260804_232300`) | every summary metric identical; out-of-fold probabilities differ by at most 3.3e-16 (floating-point rounding) | 5 min |
| `evaluate_selection.py`, CG/MBPP+ cell (`selection_mbpp_20260714_110252`) | every stored field identical; the current script also writes two extra diagnostic arms not reported in the paper | 6 min |
| `compare_regeneration_baseline.py` | identical | seconds |
| `analyze_context_mode_ladder.py`, HumanEval and MBPP base | identical | seconds |
| `analyze_smart_repair.py` | identical | seconds |
| `sweep_random_gate.py`, MBPP and HumanEval | identical | seconds |
| `diagnose_gate_target.py`, MBPP and HumanEval | identical | seconds |
| `diagnose_arm_allocation.py` | identical | seconds |
| `pilot_gate_direction.py` | identical | seconds |
| `calibrate_predictor.py`, merged-164 (`calibration_humaneval_20260713_155535`) | identical | 20 min |
| `paper/make_figures.py` | both figures regenerated | seconds |
| `compare_representations.py`, seven cells (`analysis_week7_representations_20260804_221056`) | identical | 2 h 15 min (tree edit distance over every sample pair) |

The remaining predictor, selection and calibration cells use the same scripts with
the inputs listed above. Execution features depend on a 5 s timeout, so on a much
slower or faster machine a few probe outcomes near the limit can flip (Section 5).

### Tier 2: regenerating the samples (GPU, hours)

Only needed to re-create the stored generation and repair records themselves.
Example, one rung of the capability ladder:

```bash
python scripts/run_pipeline.py --mode uncertainty --dataset humaneval \
  --n-problems 60 --n-samples 10 --temperature 0.7 --seed 42 \
  --model Qwen/Qwen2.5-Coder-3B --device cuda --dtype float32
```

Repair arms are produced by `scripts/run_iterative_repair_experiment.py`
(`--context-mode {none,code_only,error}`). The Kaggle drivers in `notebooks/` contain
the exact commands used for each GPU run. Measured wall-clock times are in
`reports/artifact_appendix.md`.

**Device and precision are part of the experiment.** Re-generating on a different
device or dtype changes the samples (for example, pass@1 moved from 0.230 to 0.2167
when one rung was regenerated on CUDA instead of CPU). Every run records
`device_resolved` and `dtype_resolved` in its config; compare like with like.

## 4. Paper element to artifact map

File names are in `results/`. Field-level detail for each cell is in
`reports/paper_number_audit.md` (its table numbers refer to an earlier draft; match
by the LaTeX label given here).

| Paper element (label) | Script | Stored result |
|---|---|---|
| Table 3, sample-level ROC-AUC (`tab:auc`) | `train_predictor.py --representation corrected` | `predictor_humaneval_20260805_000941.json`, `predictor_humaneval_20260804_232950.json`, `predictor_mbpp_20260804_232300.json`, `predictor_mbpp_20260804_233512.json` |
| Table 4, problem-level ROC-AUC (`tab:problemauc`) | same | same four files |
| Table 5, selection accuracy (`tab:selection`) | `evaluate_selection.py` | `selection_humaneval_20260714_105452.json`, `selection_humaneval_20260714_111521.json`, `selection_mbpp_20260714_110236.json`, `selection_mbpp_20260714_110252.json` |
| Table 6, prompt ablation + regeneration baseline (`tab:context`) | `run_iterative_repair_experiment.py`, `compare_regeneration_baseline.py` | `{humaneval,mbpp}_week9_arm_{resample,code_only,error_nostop}_sample0_qwen15b.json`, `analysis_regeneration_baseline_20261008_234528.json` |
| Table 7, repairs by failure category (`tab:category`) | `analyze_context_mode_ladder.py` | `analysis_week9_context_mode_ladder_humaneval_base_20260830_195949.json`, `analysis_week9_context_mode_ladder_mbpp_base_20260831_105751.json` |
| Figure 3, success under matched budgets (`fig:context`) | `paper/make_figures.py` | `paper/figure_data.json` (counts from Table 6) |
| Table 8, joint MBPP allocation (`tab:policies`) | `analyze_smart_repair.py` | `mbpp_week7_smart_repair_comparison.json` |
| Table 9, one score, two targets (`tab:targets`) | `diagnose_gate_target.py` | `analysis_week8_gate_target_20260829_211126.json`, `analysis_week8_gate_target_humaneval_20260830_100907.json` |
| Table 10, allocation signal per prompt arm (`tab:armalloc`) | `diagnose_arm_allocation.py` | `analysis_week12_arm_allocation_20260907_212130.json` |
| Figure 4, allocation under call caps (`fig:allocation`) | `pilot_gate_direction.py`, `paper/make_figures.py` | `analysis_week12_pilot_direction_20260907_222814.json` |
| Table S1, correlations before/after prompt removal (`tab:s-corr`) | `compare_representations.py` | `analysis_week7_representations_20260804_221056.json` |
| Table S2, calibration (`tab:s-calib`) | `calibrate_predictor.py` | `calibration_humaneval_20260713_153953.json`, `calibration_humaneval_20260713_155535.json` |
| Table S3, BigCodeBench (`tab:s-bcb`) | `validate_bigcodebench_cell.py`, `run_pipeline.py --dataset bigcodebench`, `analyze_uncertainty.py` | `bigcodebench_uncertainty_*.json`, `analysis_week11_bcb_*_bigcodebench_20260902_*.json`, gate `analysis_week11_bcb_gate_20260901_190205.json` |
| Table S4, held-out MBPP repair (`tab:s-heldout`) | `run_iterative_repair_experiment.py`, `analyze_smart_repair.py` | `mbpp_week6_failure_pool_seed42.json`, `mbpp_week7_smart_repair_comparison.json` (block `week6_stop_on_pass_counterfactual`) |
| Table S5, paired context comparisons (`tab:s-pairs`) | `analyze_context_mode_ladder.py`, `rescore_evalplus_labels.py --repair-results` | `analysis_week9_context_mode_ladder_humaneval_{base,plus}_*.json` |
| Table S6, random replays (`tab:s-random`) | `sweep_random_gate.py` | `analysis_week8_random_gate_sweep_20260829_211142.json`, `analysis_week8_random_gate_sweep_humaneval_20260830_100905.json` |
| Table S7, gains at call caps (`tab:s-caps`) | `pilot_gate_direction.py` | `analysis_week12_pilot_direction_20260907_222814.json` |

The generation records these analyses start from:

| Cell | Record |
|---|---|
| CodeGen-350M, HumanEval (164 problems) | `humaneval_uncertainty_merged164_20260711.json` |
| DeepSeek-Coder-1.3B, HumanEval (60) | `humaneval_deepseek60_merged_20260712.json` |
| CodeGen-350M, MBPP+ members (60) | `mbpp_uncertainty_20260712_024909.json` |
| DeepSeek-Coder-1.3B, MBPP+ members (60) | `mbpp_uncertainty_20260712_085004.json` |
| Qwen2.5-Coder ladder, HumanEval (60) | `humaneval_uncertainty_20260827_*.json`, `humaneval_uncertainty_20260831_212704.json`, `humaneval_uncertainty_20260901_004849.json` |
| Strengthened (EvalPlus) labels | `humaneval_evalpluslabel_*.json`, `*_evalpluslabel_*.json` |

## 5. Determinism

1. **Seeds.** Every entry point calls `set_all_seeds(seed)` (Python `random`, NumPy,
   PyTorch, CUDA); the default is 42. Repair candidate seeds depend only on the
   candidate index, so a resumed run is identical to an uninterrupted one.
2. **Device and dtype** are recorded in every run config (Section 3, Tier 2).
3. **Windows-style paths.** Some early records were written on Windows and store
   paths as `results\file.json`. The analysis scripts parse these with
   `PureWindowsPath`, so they reproduce on macOS and Linux.
4. **Execution timing.** Behavior signatures replay generated code with a 5 s timeout.
   Probes near that boundary can change outcome between runs. In one such re-run
   during the project, a mean execution feature moved by 8e-5 and a cross-validated
   ROC-AUC by at most 0.002. No reported conclusion depends on it.

Absolute paths that appear inside some stored records (for example in tracebacks)
are provenance strings from the machine that produced them; no script reads them.

## 6. Data and models

| Dataset | Source | Use |
|---|---|---|
| HumanEval | `openai_humaneval` (HuggingFace `datasets`) | primary benchmark |
| MBPP | `mbpp` | second benchmark; MBPP+ member subset pinned in `data/mbppplus_member_task_ids.txt` |
| HumanEval+, MBPP+ | `evalplus/humanevalplus`, `evalplus/mbppplus` | strengthened labels and probe inputs |
| BigCodeBench | `bigcode/bigcodebench` (cached copy in `data/bigcodebench/`) | harder stress-test cell; 60-problem cell pinned in `data/bigcodebench_cell_task_ids.txt` |

Each dataset remains under its original license.

Models (all open-weight): `Salesforce/codegen-350M-mono`,
`deepseek-ai/deepseek-coder-1.3b-base`, `Qwen/Qwen2.5-Coder-{0.5B,1.5B,3B,7B}`,
`Qwen/Qwen2.5-Coder-1.5B-Instruct` (the repair model), and `microsoft/codebert-base`
(embedding metrics).

## 7. Safety warning

The pipeline executes model-generated code. `src/evaluation.py` runs it in a
time-limited subprocess inside a temporary working directory, but this is **not a
sandbox**: generated code can still reach the file system and the network. Tier 1
does not execute generated code except for the behavior-signature replays in
`diagnose_crash_control.py`, `diagnose_repairability.py` and `train_predictor.py`
(execution features). Run Tier 2, and any script that replays code, inside a
container or a throwaway virtual machine.

## 8. License and citation

Code is released under the MIT License (`LICENSE`). If you use this package,
please cite the paper (see `CITATION.cff`).

Contact: Melih Elibol, melih.elibol@sabanciuniv.edu

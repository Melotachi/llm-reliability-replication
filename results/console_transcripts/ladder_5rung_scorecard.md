# 5-rung capability ladder — scorecard against the Week 7 pre-registration

**Provenance warning.** Three of these five rungs (Qwen-3B, and the codegen/deepseek
device-parity re-runs) are computed from hand-pasted console transcripts, not from run
JSONs — see `README.md` in this directory. The `initial_pass` column of each transcript
is checksum-verified against the printed `mean_pass_at_1` to 6 decimal places; the
diversity columns have no such check. **Recover the JSONs before any of this is published.**

Cell: HumanEval 0-59, N=10, temperature 0.7, seed 42, cuda/float32 for every rung
(device parity — the earlier ladder mixed CPU and CUDA rungs).

| model | pass@1 | AST \|rho\| | P3 predicted | dev | in tol | binary spread | mean-rate spread | conf-wrong | parse-fail |
|---|---|---|---|---|---|---|---|---|---|
| codegen-350M | 0.2167 | 0.601 | 0.551 | +0.050 | yes | 0.600 | 0.473 | 1/6 | 0.0150 |
| Qwen2.5-Coder-0.5B | 0.4333 | 0.627 | 0.698 | −0.071 | yes | 0.400 | 0.613 | 0/6 | 0.0333 |
| deepseek-1.3b | 0.4467 | 0.638 | 0.707 | −0.069 | yes | 0.400 | 0.560 | 0/6 | 0.0217 |
| Qwen2.5-Coder-1.5B | 0.5950 | 0.716 | 0.808 | −0.092 | yes | 0.200 | 0.627 | 0/6 | 0.0217 |
| Qwen2.5-Coder-3B | 0.6033 | 0.632 | 0.814 | **−0.182** | **NO** | 0.133 | 0.467 | 0/6 | 0.0383 |

## Verdict: 3 of 5 pre-registered predictions falsified

- **P1 (confidently-wrong residual shrinks) — HELD.** Weakly: the denominator is 6
  problems, so the rate moves only in steps of 0.167. The pre-registration said so.
- **P2 (spread stays in [0.40, 0.70], no capability trend) — FALSIFIED, decisively.**
  Two rungs fall outside the band (1.5B 0.200, 3B 0.133) and the decline is now
  statistically significant: r = −0.981, p = 0.003 over five rungs, non-increasing
  throughout. The pre-registration named this outcome in advance as "the single most
  important negative result the project could produce".
- **P3 (structural correlation strengthens with capability) — FALSIFIED at the top rung.**
  Held for the first three held-out rungs, but 3B lands 0.182 below the fitted line,
  outside the ±0.12 tolerance. |rho| is not monotone in pass@1 across five rungs.
- **P4 (parse-failure < 0.005) — FALSIFIED at every rung** (0.0150–0.0383). No capability
  trend, and no model-family explanation survives (see the report's §7.3).
- **P5 (capability is not scale) — CONFIRMED.** Qwen-0.5B (0.4333) beats codegen-350M
  (0.2167) at comparable size, which is what licenses the pass@1 x-axis.

## Two cautions before this is written up

1. **The top two rungs are barely separated in capability** (0.5950 vs 0.6033) yet differ
   by 0.084 in |rho|. At n = 60 problems that gap is plausibly estimation noise rather
   than a capability effect — consistent with the attenuation-ceiling analysis. P3's
   falsification rests on a single rung and should be reported with that caveat.
2. **The earlier "mean-rate spread rescues P2" reading does not survive the fifth rung.**
   Mean-rate spread against pass@1 is r = +0.325, p = 0.594, and non-monotone
   (0.473, 0.613, 0.560, 0.627, 0.467). Report the binary decline as the finding.

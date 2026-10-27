# Console transcripts — NOT results JSONs

These files are the *printed summary tables* of two pipeline runs, pasted by hand
from a Kaggle session whose `/kaggle/working` was lost before the JSONs could be
downloaded (2026-08-22).

**What they are:** per-problem aggregate rows — initial pass rate and the
problem-level diversity metrics that `run_pipeline.py` prints.

**What they are NOT:** the run JSONs. They contain no per-sample code, no
per-sample pass/fail, no error messages. Therefore they CANNOT feed
`analyze_capability_sweep.py`, `train_predictor.py`, `evaluate_selective_prediction.py`,
`analyze_consensus_deciles.py`, or anything else that recomputes metrics from
stored samples.

**What they CAN support:** problem-level correlations (metric vs pass rate) and
the consensus quartile table, both of which read only the columns present here.
Anything computed from them must say so explicitly.

`tree_edit_distance` was disabled on these runs (`--no-tree-edit` in the
notebook's COMMON block), so `ted_div` is nan by configuration, not by failure.
`codebert`/`bertscore` are nan for the same reason (`--no-codebert`).

Recover the real JSONs from the Kaggle notebook version's Output tab if the run
was committed with Save & Run All; an interactive session that ended without a
commit has lost them and the run must be repeated.

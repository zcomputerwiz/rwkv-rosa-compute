# rwkv-rosa-compute

Research apparatus for RWKV-7 filler-token experiments. Everything here is
operational: how to run things on this machine and what breaks. **The science
is not in this file** — see the pointers in the last section and read them when
the question is about results.

## Environment

**Always use the repo venv.** Bare `python` on this box is 3.12 with no pytest
installed; the venv is 3.13.15 with torch 2.13.0+cu126 / CUDA 12.6.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q --ignore=tests/test_exp0_cuda.py
.\.venv\Scripts\python.exe -m ruff check .
```

**Run ruff locally before pushing.** It is in the venv and CI fails on it.

**The CUDA suite needs an MSVC developer environment.** Without it, 18 tests in
`tests/test_exp0_cuda.py` fail at `verify_ninja_availability()` — this is a
shell-environment problem, not a broken build, and it looks alarming in a plain
shell. Dot-source (do not invoke) `scripts/init_cuda_env.ps1`, which resolves
the venv, locates Visual Studio via vswhere, and activates the MSVC toolset.
Ignoring that file is the right move when you only need the CPU suite; 536
tests pass without it.

## Shell traps on this machine

- **Bash heredocs fail here repeatedly** (`unexpected EOF while looking for
  matching`). Use the Write tool for any multi-line content — commit messages,
  PR bodies, scripts — or PowerShell. This has cost more time than any other
  single thing.
- **PowerShell `-Filter` takes wildcards, not regex.** `-Filter "a_[12]_*"`
  matches nothing and reports no error. Use `Where-Object {$_.Name -match ...}`.
- **Ruff E402 tolerates a bare `sys.path` mutation before imports and nothing
  else.** Adding any other statement in that position — even an assignment —
  makes every import below it an error. Scripts here rely on that allowance to
  reach `src/`; put derived constants *after* the imports.

## GPU rules

- **One CUDA process at a time.** Two concurrent jobs doubled wall clock from
  329 s to 695 s and made the runs non-comparable with their references.
- **cudagraphs requires every batch the same shape.** Memory counts must be
  multiples of `batch_size / gcd(batch_size, queries_per_memory)` — 64 at the
  usual batch 256 with 4 queries per memory. A ragged bank is fatal under
  cudagraphs, not merely slow, and it raises partway through the first
  evaluation after training has already run. The audit tool refuses up front.
- **Oversubscription is not OOM.** Over-large batches spill to host memory and
  report success while throughput drops ~20×. Bound sweeps by measured
  throughput, not by whether they crash.

## Artifacts and provenance

- Use `get_artifact_environment()` from `rosa_compute.diagnostics`, never
  `get_environment_info()` directly: the latter returns a `BuildCapabilities`
  object that `json.dump` cannot serialise, and filtering it to scalars
  silently drops `gpu_compute_capability` — the one field that distinguishes
  sm_75 / sm_86 / sm_89.
- Artifacts record full commit SHA, tree hash and dirty flag. **Merge before
  dispatching fleet work.** Pointing other nodes at a live branch produced
  artifacts naming a SHA that stopped resolving once the branch squash-merged
  and was deleted.
- **Report `held_out_final`, never `held_out_best`.** The gates decide on the
  final epoch; a maximum over 32–128 epochs is a selection presented as an
  outcome.

## The fleet

Coordination happens through `D:\ProjectSync` (Syncthing). Publish every
document and artifact with a sha256 sidecar — `<hash><two spaces><filename>`
plus a trailing CRLF — and verify others' sidecars against their files before
trusting a reported number. Read the JSON, not the summary table.

```text
claude-ada        RTX 4060 Ti   sm_89   330 s   this machine; chipset-attached
                                                B550 Gen3 x4; secondary GPU
gemini-turing     RTX 2070      sm_75   423 s   CPU-direct 3.0 x16; drives display
antigravity-ampere RTX 3070 Lp  sm_86   506 s   laptop; iGPU drives display
codex-shannon                                   auditor, no GPU
opencode-dijkstra                               CPU only
```

Times are for the standard cell (19,968 memories, 32 epochs, batch 256,
compiled). Same seed converges on one card and not another — this is
reproducible, not noise. sm_89 shows occasional run-to-run numerical
divergence (4 of 12 repeats); the other two have been bitwise stable.

## Standing constraints

Verbatim, from the operator:

- Do not trade experimental fidelity for throughput.
- Do not call length grouping "compute matching".
- Do not adopt without explicit separate approval: L2Wrap, altered CE
  objective, FP8, quantized weights, reduced output vocabulary, BF16 recurrent
  state if current semantics use FP32.
- Do not loosen tolerances merely because a new kernel fails them.
- Do not make TF32 the silent default for "FP32".

Gates are stopped unless a document says otherwise. Nothing re-runs,
re-specifies or reorders a gate without `codex-shannon`'s ruling.

## Where things are

```text
docs/development.md                  repo rules: upstream isolation, CPU oracle,
                                     CI strategy. Read before changing src/.
docs/experiments.md                  research sequence H1-H4, protocols
docs/experiment1_recall_capacity.md  current experiment: gate 0a, the recall
                                     audit, and the limits of what it shows
docs/experiment0_cuda.md             CUDA build requirements and gates
scripts/audit_recall_capacity.py     the analysis tool; not a gate
scripts/run_rate_campaign.py         pre-registered rate campaign driver
scripts/init_cuda_env.ps1            dot-source for MSVC + venv
```

The outcome at the current cell is **bimodal** — runs land at ceiling or at
chance with nothing between — so the estimand is a success *rate* with a
binomial interval, not a mean or a median. If you find yourself computing a
median over three seeds, read `docs/experiment1_recall_capacity.md` §5b first.

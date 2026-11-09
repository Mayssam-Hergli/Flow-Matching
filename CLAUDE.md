# CLAUDE.md — Flow Matching for circular antenna array design

This file is the **architecture contract** of the project. Read it fully at the start of every session and follow it.
If a request conflicts with this file, stop and say so instead of guessing.

## 1. Project in two lines
Inverse design of a 5-ring concentric circular antenna array (INSAT, Pr. Naoufel Ismail, 2026/2027).
Given specs `[G, SSL, HPBW, theta0]`, a **conditional Flow Matching** model generates several candidate
architectures `[N1..N5]`; each candidate is verified with the professor's physics code and the best one is selected.
The method is described in `rapport/rapport_flow_matching.pdf` (sections 3 and 6).

## 2. Professor's rules (non-negotiable)
1. Exactly **four separate scripts** in `livrable/`: dataset, train/test split, training, test.
2. Radiation pattern, Gmax, SSL, HPBW and prediction error are computed with **the professor's code**
   (`calcul_AF_performance_metrics` and the error formulas of `prof/test_model_FFNN.py`), copied **verbatim**.
   Never change, "fix", optimize or vectorize them.
3. Framework: **PyTorch** only.
4. Every script reads and writes its files **in its own folder**:
   `script_dir = os.path.dirname(os.path.abspath(__file__))`.
5. Matrices produced by the dataset script and the split script are saved in **both `.npy` and `.mat`**.
6. The professor runs the scripts himself with `python <script>.py`, in order, with no edits. No notebooks, no Colab.

## 3. Repository layout
```
FlowMatching/
├── CLAUDE.md              ← this file (architecture contract)
├── JOURNAL.md             ← what was done at each step, decisions, results (append only)
├── README.md              ← for humans (French): what the project is, how to run it
├── requirements.txt
├── .gitignore
├── .claude/settings.json  ← Claude Code settings (no AI attribution in commits)
├── prof/                  ← the professor's 5 original scripts. NEVER MODIFY.
├── livrable/              ← the 4 delivered scripts (+ final trained files, committed at the end)
│   ├── dataset_FM.py
│   ├── repartition_training_test_FM.py
│   ├── FM_training.py
│   └── test_model_FM.py
├── outils/                ← dev-only helpers (sanity checks, consistency checks). Not delivered.
└── rapport/               ← LaTeX report
```

## 4. Pipeline and file contract
Scripts communicate **only through files** in `livrable/`. Names, shapes and keys below are fixed.

| Script | Reads | Writes |
|---|---|---|
| `dataset_FM.py` | — | `Minput.npy/.mat` (key `Minput`, shape **4 × N**), `Moutput.npy/.mat` (key `Moutput`, shape **5 × N**) |
| `repartition_training_test_FM.py` | `Minput.npy`, `Moutput.npy` | `Minput_training`, `Moutput_training`, `Minput_test`, `Moutput_test` in `.npy` and `.mat` (`.mat` key = variable name) |
| `FM_training.py` | `Minput_training.npy`, `Moutput_training.npy` | `fm_model.pth`, `fm_normalization_params.npz`, `fm_config.json`, `fm_courbe_loss.png` |
| `test_model_FM.py` | `Minput_test.npy`, `Moutput_test.npy`, `Minput.npy`, `fm_*` files | figures `fm_*.png`, `fm_resultats.txt`, `fm_resultats.npy/.mat` |

**Matrix conventions (same as the professor):**
- `Minput` rows: `[0] G` main lobe gain in dB, non-normalized (`20·log10 max|AF|`);
  `[1] SSL` in dB relative to the main lobe (≤ 0); `[2] HPBW` in degrees; `[3] theta0` in degrees, in [0, 180].
- `Moutput` rows: `N1..N5`, number of elements per ring, integers 0..10 stored as floats. All-zero is invalid.
- Files store **features × samples** (columns = samples). Scripts transpose to **samples × features** when loading for PyTorch.
- Split: 90 % training / 10 % test, seed `46`, exactly like the professor. Validation (10 % of training) is carved out inside `FM_training.py`.
- Seeds fixed to 46 for **both** NumPy (`np.random.seed(46)`) and PyTorch (`torch.manual_seed(46)`) in every script we write,
  so every run is reproducible. The professor's FFNN does not fix the torch seed, so it is not reproducible:
  for comparisons it must be trained several times and reported as mean ± std.

**Our files never overwrite the professor's:** every file we create starts with `fm_`
(e.g. `fm_normalization_params.npz`, not `normalization_params.npz`).

## 5. Model conventions (from the report, section 6)
- Architecture scaling: `x = 2·N/10 − 1` ∈ [−1, 1]; back to integers: invert, round, clip to [0, 10].
- Dequantization noise during training only: U(−0.1, 0.1).
- Condition: `c = [G, SSL, HPBW, sin(theta0)]`, standardized with training mean/std
  (the pattern depends on theta0 only through sin(theta0)).
- Path: `x_t = (1−t)·x0 + t·x1`, `x0 ~ N(0, I)`, target velocity `x1 − x0`, MSE loss.
- Velocity network (final version, report fig. 5–6 and table 2):
  input projection 5→256; time coding (64 sinusoids + dense 64→64); condition MLP 4→64→64;
  context = concat (128); 4 residual blocks of width 256 (LayerNorm → FiLM from context → dense → SiLU → dense, + skip);
  output LayerNorm + dense 256→5. About 8×10⁵ parameters.
  Classifier-free guidance: condition replaced by a learned null vector 10 % of the time during training.
- Inference: generate K candidates (K = 64, Heun, 20 steps, guidance w) → back to integers, remove duplicates and all-zero
  → verify each with the professor's function → select the lowest **professor's global weighted error**
  (`error_pred_global`, weights 0.33/0.33/0.34) computed **against the requested specs**.
- Every hyperparameter lives in one place: a block at the top of `FM_training.py`, saved to `fm_config.json`.
  `test_model_FM.py` rebuilds the model **from `fm_config.json`**, never from hard-coded sizes.

## 6. Shared code
Several scripts need the same code (physical parameters, the professor's metric function, the model class).
Like the professor's own code, each script stays **self-contained**: shared code is **copied**, not imported.
- Each copied block is wrapped in
  `# ===== BLOC COMMUN : <nom> — identique dans <scripts> — ne pas modifier ici seul =====` … `# ===== FIN BLOC COMMUN =====`.
- `outils/verifier_blocs_communs.py` checks that all copies of each block are byte-identical. Run it before every commit.
- Physical parameter block (identical in every script):
  `carrierFreq = 2.45e9`, `c = 3e8`, `lambda_ = c / carrierFreq`, `r0 = 0.2 * lambda_`, `delta_r = 0.5 * lambda_`,
  `max_rings = 5`, `max_elements = 10`.

## 7. Code style
- Comments and console prints **in French**, explanatory, in the professor's style (`clear_console` banner, etc.).
- Plots: `plt.savefig(...)` into `script_dir`, then `plt.show()` like the professor.
  During development run headless (`$env:MPLBACKEND="Agg"` in PowerShell).
- No extra frameworks, no notebooks. Dependencies only in `requirements.txt`.
- Environment: Windows, PowerShell.

## 8. How to work (rules for Claude Code)
- Do **only the step you are asked for**. Never start a later step or add features that were not requested.
- Install any dependency you need without asking. If something fails, debug and fix it yourself.
- Never modify anything in `prof/`. Never modify this file unless explicitly asked (except the status column of section 9).
- **Git:**
  - Never add `Co-Authored-By`, "Generated with Claude Code", session links or any other AI attribution
    to commit messages or PR descriptions.
  - Make **one commit** per step with message `etape N : <résumé>`, authored with the existing git config. **Do not push**:
    the human checks the commit and pushes herself.
  - Never commit generated files (`*.npy`, `*.mat`, `*.pth`, `*.npz`, `*.png`) unless explicitly asked (step 5).
- At the end of each step:
  1. run the step's checks and report the real output (numbers, shapes, errors);
  2. append a section to `JOURNAL.md`: what was done, files created/changed, decisions taken and why, results, open questions;
  3. run `outils/verifier_blocs_communs.py` (once it exists);
  4. update the status column of section 9 (☐ → ☑);
  5. make the commit (no push);
  6. finish with a short summary in French for the human: what was done, how to check it, what is next.
- Explain decisions in plain words. The human must understand everything that is committed.

## 9. Steps and status
| Step | Content | Status |
|---|---|---|
| 0 | Repo skeleton, README, requirements, .gitignore, JOURNAL; run the professor's FFNN pipeline unchanged (baseline) | ☑ |
| 1 | `dataset_FM.py` + `repartition_training_test_FM.py` (small N first) + physics sanity check | ☑ |
| 2 | Crude `FM_training.py` (no OT, no guidance, no EMA) | ☐ |
| 3 | Crude `test_model_FM.py`: generate → verify → select, first FM vs FFNN numbers | ☐ |
| 4 | Improvements one at a time, each measured: Heun, guidance, OT pairing, EMA, N_tot balancing, 50k samples, final network size | ☐ |
| 5 | Final: French README with run instructions, clean run in a fresh clone, commit final trained files with `git add -f` | ☐ |

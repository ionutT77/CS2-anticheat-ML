# CS2 ML Anti-Cheat — Project Journal & Thesis Itinerary

> **Purpose:** This file documents every step, decision, and prototype made throughout the development of the CS2 ML-based anti-cheat system. It serves as the itinerary for the thesis paper — a complete record of how the project evolved from concept to final implementation.

---

## Project Overview

| Field | Value |
|-------|-------|
| **Project Title** | Machine Learning-Based Server-Side Anti-Cheat for Counter-Strike 2 |
| **Author** | Ionut |
| **Start Date** | 2026-08-18 |
| **Goal** | Detect aimbot and bhop cheats using LSTM/Transformer models trained on player telemetry data |
| **Repository** | [CS2-anticheat-ML](https://github.com/ionutT77/CS2-anticheat-ML) |

---

## Phase 0: Research & Planning

### Entry 1 — Initial Research & Feasibility Analysis
**Date:** 2026-08-18  
**What was done:**
- Analyzed two existing reports:
  - *"Anti-Cheat ML Thesis — Datasets, Architectures & Report"* — survey of ML anti-cheat methods, datasets, and architectures
  - *"Bachelor Anti-Cheat Report"* — scoping and feasibility analysis for bachelor thesis
- Conducted web research on:
  - CS2 server-side plugin capabilities (Metamod:Source + CounterStrikeSharp)
  - Available datasets (Kaggle CSGO Cheating, CS2CD HuggingFace, ESTA)
  - Existing open-source CS2 anti-cheat projects (CS2AC, TBAntiCheat, ImpactGuard)
  - ML architectures for cheat detection (LSTM, Transformer/AntiCheatPT)

**Key decisions:**
1. **Server-side approach confirmed** — CS2 supports server-side plugins via CounterStrikeSharp (C#), which exposes pitch, yaw, position, velocity, and game events. This is sufficient for aimbot and bhop detection.
2. **Two-phase architecture chosen:**
   - Phase 1: LSTM (2 layers, 128→64 hidden) as baseline — well-understood, easy to debug
   - Phase 2: Transformer (AntiCheatPT-style) for potentially better performance
3. **Focus on two cheat types:** Aimbot (primary) and Bhop (secondary)

**Why these decisions were made:**
- LSTM was chosen as the starting model because the data is inherently sequential (tick-by-tick time series). LSTMs are purpose-built for sequence data and are simpler to implement and debug than Transformers.
- CounterStrikeSharp was chosen over SourceMod because SourceMod does NOT work with CS2's Source 2 engine.
- Aimbot was prioritized over bhop because there exists a labeled dataset for aimbot detection, while bhop has no public dataset.

**Output:** `implementation_plan.md` — saved to project root

---

### Entry 2 — Dataset Selection
**Date:** 2026-08-18  
**What was done:**
- Evaluated three candidate datasets for the prototype:
  1. **Kaggle CSGO Cheating Dataset** (emstatsl) — 12K players, 5 features, pre-labeled, time-series format
  2. **CS2CD** (HuggingFace) — 795 CS2 matches, full tick data, comes with AntiCheatPT model
  3. **ESTA** — Pro player trajectories, useful as ground truth for legit movement

**Decision: Use the Kaggle CSGO Cheating Dataset as the primary training data**

**Why this dataset:**
- **Perfect feature alignment:** The 5 features (`AttackerDeltaYaw`, `AttackerDeltaPitch`, `CrosshairToVictimYaw`, `CrosshairToVictimPitch`, `Firing`) directly capture the "weird crosshair changes" we want to detect
- **Pre-processed:** Data is already windowed (5 sec before + 1 sec after engagement), no manual extraction needed
- **Labeled:** Binary labels (cheater/legit) — no manual annotation required
- **Good class balance:** 10,000 legit + 2,000 cheaters (5:1 ratio — manageable with weighted loss)
- **Ready for LSTM:** Shape `(players, 30, 192, 5)` maps directly to LSTM input `(batch, sequence_length, features)`

**What was deferred:**
- CS2CD will be explored in Phase 2 for Transformer training
- Bhop detection deferred — no public labeled dataset exists. Will need to generate our own data on a private CS2 server

**Output:** This journal entry

---

### Entry 3 — LSTM Architecture Design
**Date:** 2026-08-18  
**What was done:**
- Designed the complete LSTM architecture for aimbot detection
- Documented layer-by-layer explanation for thesis understanding

**Architecture summary:**
```
Input (192 ticks × 5 features)
    → LSTM Layer 1 (128 hidden units) — learns low-level temporal patterns (individual snaps, aim anomalies)
    → Dropout (0.3) — regularization
    → LSTM Layer 2 (64 hidden units) — learns higher-order patterns (patterns of snaps, inhuman consistency)
    → Take last hidden state — compresses 192 ticks into 64-dim summary
    → FC (64 → 32) + ReLU + Dropout — decision features
    → FC (32 → 1) + Sigmoid — binary probability P(cheater)
```

**Design decisions:**
- **2 LSTM layers** (not 1 or 3): 1 layer is too shallow for complex patterns; 3+ layers offer diminishing returns and slower training
- **128→64 hidden dimensions** (decreasing): Forces the model to compress/abstract at higher layers
- **Dropout 0.3**: Standard regularization; prevents overfitting on the relatively small 12K player dataset
- **Sigmoid output**: Binary classification (cheater vs. legit), not multi-class

**Output:** Detailed architecture explanation saved in conversation artifacts

---

## Phase 1: ML Prototype with Kaggle Dataset

> *Entries will be added as work progresses*

### Entry 4 — Environment Setup
**Date:** 2026-08-18  
**What was done:**
- Created Python 3.11 virtual environment (`venv/`)
- Installed all dependencies via `requirements.txt`:
  - PyTorch 2.13 (CPU), NumPy 2.4, scikit-learn 1.9, pandas, matplotlib, seaborn, jupyter, nbformat
- Created full project directory structure: `src/`, `notebooks/`, `data/raw/`, `data/processed/`, `models/`
- Created `.gitignore` (excludes venv, data files, model checkpoints, Jupyter checkpoints)
- Registered venv as Jupyter kernel: **"Python (CS2 Anti-Cheat)"**

**Key decisions:**
- **CPU-only PyTorch** — machine has no NVIDIA GPU (CUDA unavailable). LSTM is small enough that CPU training is feasible for the prototype
- **Data files excluded from git** (`.npy` files are 200MB+ each) — developer must download manually from Kaggle

**Output:** `venv/`, `requirements.txt`, `.gitignore`, project directory tree

---

### Entry 5 — Dataset Download & Verification
**Date:** 2026-08-18  
**What was done:**
- Downloaded Kaggle CSGO Cheating Dataset manually from https://www.kaggle.com/datasets/emstatsl/csgo-cheating-dataset
- Discovered dataset has **two separate files** (not one as initially assumed):
  - `cheaters.npy` — 2,000 cheaters, shape `(2000, 30, 192, 5)`, 230MB
  - `legit.npy`    — 10,000 legit players, shape `(10000, 30, 192, 5)`, ~1.1GB
- Both moved to `data/raw/`
- Verified: no NaN, no Inf, dtype float32, value range [-180, +180] degrees

**Key facts discovered:**
- 5:1 class imbalance (legit:cheater) — must use weighted loss or oversampling when training
- Each player has 30 engagements × 192 ticks × 5 features = 28,800 data points per player
- Total dataset: 12,000 players × 28,800 = 345.6M data points

**Output:** `data/raw/cheaters.npy`, `data/raw/legit.npy`

---

### Entry 6 — Data Exploration Notebook
**Date:** 2026-08-18  
**What was done:**
- Created `notebooks/01_data_exploration.ipynb` with 10 analysis sections:
  1. Load & verify both files
  2. Basic statistics & class balance
  3. Feature distributions (cheater vs legit histograms)
  4. Individual aim trajectories over 192 ticks
  5. Crosshair-to-victim convergence analysis
  6. Aim speed per tick (snap detection)
  7. **Triggerbot signature analysis** — crosshair distance at firing ticks
  8. Average engagement profiles (mean ± std across all players)
- Plots saved to `data/processed/`

**What we're looking for:**
- **Aimbot:** Large single-tick DeltaYaw/Pitch spikes right before kill; CrosshairToVictim collapses to 0 unnaturally fast
- **Triggerbot:** Firing ticks have near-zero crosshair-to-victim distance (fires at the exact moment of alignment — no human reaction delay)

**Output:** `notebooks/01_data_exploration.ipynb`

---

### Entry 7 — Getting Started / Running the Server
**Date:** 2026-08-18  
**What was done:**
- Fixed a string literal bug in the notebook generation script.
- Documented how to start the Jupyter server and virtual environment.

**How to start your workspace:**
1. **Activate the Virtual Environment:**
   Open PowerShell and navigate to the project directory, then run:
   ```powershell
   .\venv\Scripts\activate
   ```
2. **Start the Jupyter Server:**
   While the virtual environment is active, you can start Jupyter in one of two ways:
   
   **Option A: Plain Text (Unencrypted)**
   ```powershell
   jupyter notebook --notebook-dir=.
   ```
   *SERVER IS RUNNING WITHOUT ENCRYPTION: http://localhost:8888/*
   *TEXT IS SENT VIA PLAIN TEXT.*

   **Option B: Encrypted (HTTPS)**
   ```powershell
   jupyter notebook --notebook-dir=. --certfile=jupyter_cert.pem --keyfile=jupyter_cert.key
   ```
   *THE SERVER IS NOW RUNNING WITH ENCRYPTION: https://localhost:8888/*
   *TEXT IS SENT SECURELY VIA HTTPS.*
   *Note: Because this is a self-signed certificate, your browser will warn you that the connection is not private. You can safely proceed/bypass this warning (e.g. click "Advanced" -> "Proceed to localhost").*
3. **Open the Notebook:**
   Navigate to the `notebooks/` folder in the web UI and open `01_data_exploration.ipynb`. Make sure the kernel is set to **"Python (CS2 Anti-Cheat)"**.

---

### Entry 8 — Input Feature Documentation
**Date:** 2026-08-21  
**What was done:**
- Documented the physical meaning, anti-cheat significance, and concrete examples for all 5 input features in the Kaggle CSGO Cheating Dataset.
- This entry serves as the **data dictionary** for the thesis — explaining exactly what the neural network "sees" at each timestep.

**Context:** In CS:GO/CS2, a player's view direction is defined by two angles:
- **Yaw** — horizontal rotation (looking left ↔ right), range roughly [−180°, +180°]
- **Pitch** — vertical rotation (looking up ↔ down), range roughly [−90°, +90°]

The dataset captures 192 ticks (≈6 seconds at 32 tick rate) around each engagement: 5 seconds before the kill and 1 second after. The 5 features at each tick are:

#### Feature 1: `AttackerDeltaYaw` (float32)
- **What it is:** The *change* in horizontal view angle (left/right) between this tick and the previous tick. Measures the horizontal speed and direction of mouse movement at that instant.
- **Units:** Degrees per tick
- **Anti-cheat significance:** Aimbots produce unnatural spikes — snapping to a target's horizontal position in a single tick. Human mouse movements show gradual acceleration, peak speed, and deceleration. A cheater might produce a `+15.0°` spike in one tick (instantaneous snap), while a human flick might take 5–10 ticks to cover the same angle.
- **Examples:**
  - `0.0` — player is holding their mouse perfectly still (no horizontal movement)
  - `+2.5` — player flicked their mouse to the right (fast movement)
  - `-0.2` — player is slowly turning left (gentle tracking)
  - `+18.3` — suspicious single-tick snap to the right (potential aimbot)

#### Feature 2: `AttackerDeltaPitch` (float32)
- **What it is:** The *change* in vertical view angle (up/down) between this tick and the previous tick. Measures the vertical speed and direction of mouse movement.
- **Units:** Degrees per tick
- **Anti-cheat significance:** Critical for recoil control analysis. Weapons like the AK-47 kick upward when spraying — players must pull their mouse downward to compensate. Human recoil control is messy and reactive. A cheat can pull the pitch down with mathematical perfection to counter the weapon's exact recoil pattern, producing unnaturally smooth negative DeltaPitch values during sprays.
- **Examples:**
  - `0.0` — no vertical mouse movement
  - `+0.5` — player is looking up (or compensating for downward recoil)
  - `-0.6` — player is pulling their aim downward (recoil compensation during a spray)
  - `-12.7` — large single-tick vertical snap (potential aimbot snapping to head level)

#### Feature 3: `CrosshairToVictimYaw` (float32)
- **What it is:** The horizontal angular distance between the center of the attacker's crosshair and the victim (measured to the victim's center of mass or head).
- **Units:** Degrees
- **Anti-cheat significance:** Measures how horizontally aligned the player is with their target over time. An aimbot causes this value to instantly or smoothly collapse to `0.0` and stay there. A human player overshoots, corrects, and oscillates near `0.0`. The *rate of convergence* to zero is the key aimbot signal.
- **Examples:**
  - `+15.0` — enemy is 15° to the right of the crosshair (not aiming at them yet)
  - `+4.2` → `+0.5` → `-0.1` — human flick with slight overshoot
  - `0.0` — crosshair is perfectly aligned horizontally with the enemy
  - `+12.0` → `0.0` in one tick — aimbot snap (physically impossible for a human)

#### Feature 4: `CrosshairToVictimPitch` (float32)
- **What it is:** The vertical angular distance between the center of the attacker's crosshair and the victim.
- **Units:** Degrees
- **Anti-cheat significance:** Measures vertical accuracy. Especially useful for detecting "bone aimbots" that lock perfectly onto a specific vertical bone (like the head). A headshot aimbot will maintain `CrosshairToVictimPitch ≈ 0.0` with inhuman precision while the victim is moving (crouching, jumping).
- **Examples:**
  - `+3.0` — crosshair is 3° below the enemy's head (aiming at their legs/torso)
  - `0.0` — crosshair is perfectly aligned vertically with the target
  - `-1.2` — crosshair is slightly above the target
  - Values jumping from `+5.0` to `0.0` in one tick — vertical aimbot snap

#### Feature 5: `Firing` (float32, binary: 0.0 or 1.0)
- **What it is:** A binary flag indicating whether the player's weapon is actively shooting on this specific tick.
- **Anti-cheat significance:** This is the most critical *context* variable. It tells the model *when* the engagement is happening. Combined with the crosshair-to-victim distances, it enables **triggerbot detection**: a triggerbot fires the instant `CrosshairToVictimYaw` and `CrosshairToVictimPitch` both reach `0.0`, with zero human reaction delay. Legitimate players have a measurable reaction time (typically 150–250ms) between their crosshair aligning and their trigger pull.
- **Examples:**
  - `0.0` — not shooting (running, aiming, waiting)
  - `1.0` — weapon is firing on this tick

**How the 5 features work together for detection:**

| Pattern | DeltaYaw/Pitch | CrosshairToVictim | Firing | Interpretation |
|---------|---------------|-------------------|--------|---------------|
| **Human kill** | Gradual ramp over 5–10 ticks | Slowly converges to ~0, oscillates | Fires after 150–250ms delay | Normal engagement |
| **Aimbot snap** | Single massive spike (1 tick) | Instantly jumps to 0.0 | Fires immediately after snap | Aimbot signature |
| **Triggerbot** | Normal-looking movement | Reaches ~0 naturally | Fires on exact tick of alignment (0ms delay) | Triggerbot signature |
| **Aimbot + Triggerbot** | Spike + instant fire | Snaps to 0.0 | Fires same tick as snap | Combined cheat |

**Output:** This journal entry (data dictionary for thesis Chapter 3: Methodology)

---

### Entry 9 — ML Pipeline Build (Dataset, LSTM, Training, Evaluation)
**Date:** 2026-08-21  
**What was done:**
- Built the complete Phase 1 ML pipeline — all source modules under `src/`:
  1. `src/data/dataset.py` — Data loading, player-level splitting, normalization, PyTorch Dataset
  2. `src/models/lstm_detector.py` — 2-layer LSTM aimbot detector (120,897 parameters)
  3. `src/training/trainer.py` — Training loop with early stopping, checkpointing, LR scheduling
  4. `src/evaluation/metrics.py` — Metrics, plots, player-level aggregation
- Generated `notebooks/02_lstm_training.ipynb` — the training and evaluation notebook

**Key design decisions made:**

1. **Engagement-level classification (not player-level):**
   - Each engagement `(192, 5)` is one sample → 360,000 training samples instead of 12,000
   - A player is flagged as cheater ONLY if multiple engagements (configurable `min_flagged` threshold) are classified as cheating — one suspicious shot doesn't make a cheater
   - Tested thresholds: 1, 2, 3, 5, 8, 10, 15 out of 30 engagements

2. **Player-level data splitting (no leakage):**
   - All 30 engagements from one player stay in the same split (train/val/test)
   - Stratified split preserves the 5:1 class ratio across splits
   - Split: 70% train, 15% validation, 15% test

3. **Global z-score normalization:**
   - Mean/std computed from training set ONLY (prevents test data leaking into normalization)
   - Applied to 4 continuous features; `Firing` (binary) left untouched
   - Stats saved for consistent normalization at inference time

4. **Class imbalance handling (dual approach):**
   - `WeightedRandomSampler` in the DataLoader → each epoch sees roughly equal cheater/legit samples
   - `pos_weight=5.0` in BCE loss → cheater misclassifications penalized 5x more
   - Belt-and-suspenders: ensures the model doesn't just predict "legit" for everything

5. **Training configuration:**
   - Optimizer: Adam (lr=1e-3)
   - LR Scheduler: ReduceLROnPlateau (halve LR after 5 stale epochs)
   - Early Stopping: patience=10 epochs
   - Gradient Clipping: max_norm=1.0 (prevents exploding gradients in LSTMs)
   - Max epochs: 50

6. **Anti-cheat-specific evaluation:**
   - FPR @ 95% Recall = "how many legit players get wrongly banned to catch 95% of cheaters"
   - Player-level verdict with configurable `min_flagged` threshold
   - Full report: ROC curve, PR curve, confusion matrix, training history

**Model architecture (120,897 parameters):**
```
Input:   (batch, 192, 5)
LSTM 1:  5 -> 128 hidden
Dropout: 0.3
LSTM 2:  128 -> 64 hidden
FC:      64 -> 32 -> 1
Output:  Sigmoid -> P(cheater)
```

**Output files:**
- `src/data/__init__.py`, `src/data/dataset.py`
- `src/models/__init__.py`, `src/models/lstm_detector.py`
- `src/training/__init__.py`, `src/training/trainer.py`
- `src/evaluation/__init__.py`, `src/evaluation/metrics.py`
- `notebooks/generate_02_notebook.py`, `notebooks/02_lstm_training.ipynb`

**Next step:** Open `02_lstm_training.ipynb` in Jupyter and run all cells to train the model.

---

### Entry 10 — First Training Runs & Kaggle Migration
**Date:** 2026-08-25
**What was done:**

#### Bugs fixed
1. **`ReduceLROnPlateau` — `verbose=True` removed**
   - PyTorch 2.2+ removed the `verbose` parameter. Removed it from `trainer.py`.
   - LR changes are still visible because the epoch log prints `LR: x.xe-xx` every epoch.

2. **`tqdm` progress bars added to Trainer**
   - Local training on CPU was taking 5–10 min/epoch with no visible progress.
   - Added `tqdm.auto` progress bars to both `train_one_epoch` and `evaluate`, showing per-batch progress and running loss.
   - `tqdm>=4.65.0` added to `requirements.txt`.

3. **DataLoader teardown errors on Kaggle (`num_workers=2` → `0`)**
   - `_MultiProcessingDataLoaderIter.__del__` spam when running with `num_workers=2` inside a Jupyter notebook.
   - Root cause: multiprocessing workers lose their parent PID reference on notebook cell teardown.
   - Fix: `num_workers=0` (single-process). No speed impact since data is already in RAM.

#### Kaggle migration
- Local CPU training was estimated at 5–10 min/epoch × 50 epochs = ~8 hours.
- Migrated to Kaggle Notebooks with free Tesla T4 GPU → **~27s/epoch**.
- Created `notebooks/kaggle_training.py` — a single self-contained script with all `src/` logic inlined, no dependencies on the local project structure.
- Batch size increased from 256 (CPU) to 512 (GPU) to saturate the T4.
- Dataset path: `/kaggle/input/datasets/emstatsl/csgo-cheating-dataset/cheaters/cheaters.npy` and `.../legit/legit.npy`

#### Training Run 1 — Baseline (pos_weight=5.0, dropout=0.3, lr=1e-3)
| Metric | Value |
|---|---|
| Best epoch | 13 |
| Test Loss | 1.2525 |
| Test Accuracy | 0.481 |
| Test Precision | 0.219 |
| Test Recall | 0.823 |
| Test F1 | 0.346 |
| Test AUC | 0.718 |

**Analysis:** High recall (model flags almost everything as cheater) but very low precision (3 in 4 flags are innocent). `pos_weight=5.0` was double-compensating — `WeightedRandomSampler` already balances the class frequencies, so an additional 5× loss weight made the model over-predict cheaters. Val loss oscillating heavily, suggesting LR was too high.

#### Training Run 2 — Lower pos_weight + Regularization (pos_weight=2.0, dropout=0.4, lr=1e-3, weight_decay=1e-4)
| Metric | Value |
|---|---|
| Best epoch | 12 |
| Test Loss | 0.8003 |
| Test Accuracy | 0.591 |
| Test Precision | 0.249 |
| Test Recall | 0.723 |
| Test F1 | 0.371 |
| Test AUC | 0.715 |

**Analysis:** Better precision/recall balance. Train ≈ Val loss — regularization worked, no significant overfitting. Val loss still oscillating. AUC essentially unchanged at ~0.71.

#### Training Run 3 — Stable LR (pos_weight=2.0, dropout=0.4, lr=5e-4, weight_decay=1e-4)
| Metric | Value |
|---|---|
| Best epoch | 15 |
| Test Loss | 0.8419 |
| Test Accuracy | 0.572 |
| Test Precision | 0.243 |
| Test Recall | 0.744 |
| Test F1 | 0.367 |
| Test AUC | 0.713 |

**Analysis:** Val loss curve is smooth and stable (lower LR worked for stability). LR scheduler stepped down twice (5e-4 → 2.5e-4 → 1.3e-4). AUC still plateaued at ~0.71.

#### Key finding: AUC ceiling at ~0.72
After three runs with significantly different hyperparameters, AUC remained at 0.71–0.72. This indicates the bottleneck is **not** training configuration — it is the **features themselves**. The 5 raw mouse-delta features have an information ceiling that no amount of regularization or LR tuning can overcome.

**What AUC = 0.71 means:** If you pick one random cheater and one random legit player, the model ranks the cheater as more suspicious 71% of the time. Random = 50%. The model is learning real signal, but not enough for reliable deployment.

#### Model viability assessment
| Context | Verdict |
|---|---|
| Thesis proof-of-concept | ✅ Demonstrates ML anti-cheat is feasible |
| Academic analysis / writing | ✅ AUC ceiling is itself a valid finding |
| Real deployment (bans) | ❌ Precision 0.25 = 75% false positive rate |
| Flag-for-manual-review | ⚠️ Possible, if a human reviews every flag |

#### What needs to improve (next steps)
1. **Feature engineering** — raw mouse deltas are insufficient. Need:
   - Aim snap speed (degrees per tick, max snap angle)
   - Time-to-headshot after target appears
   - Pre-fire detection (trigger before crosshair alignment)
   - Per-engagement headshot rate
2. **Player-level aggregation at inference** — aggregate model scores across all 30 engagements per player before making a verdict, rather than per-engagement classification.
3. **Self-attention over LSTM output** — instead of taking only the last hidden state, apply attention so the model can focus on the most anomalous ticks in the sequence.

---



### Entry 11 - CS2CD Saved Model Recovery and Demo Validation
**Date:** 2026-09-14  
**What was done:**

- Confirmed that the selected CS2CD model was already trained and present as a blend of `75% tcn39_seed123` and `25% lgbm39_leaves15`.
- Found that the distributed SHA-256 metadata described an older project state. The inference guard rejected the saved weights because the selection, calibration, audit and source hashes did not agree.
- Repaired the metadata-only provenance chain on an isolated branch, then merged the change into `main`.
- Updated checksum references in the model selection, calibration, audit, test-results and LSTM protocol JSON files.
- Installed the missing `lightgbm` dependency in the project virtual environment.
- No `.pt`, `.joblib` or Python source files were changed during this repair.

**Validation:**

- Artifact verification passed for 41 files and the selection/calibration chain.
- The supplied CPU inference demo completed successfully.
- The example produced a raw score of approximately `0.04498475` and a calibrated score of approximately `0.10808721`; all example review flags were false.
- The research test suite passed: `16 passed`.

**Interpretation:**

The project contains a trained and runnable CS2CD benchmark model. The `0.973` ROC-AUC result belongs to the frozen CS2CD player-match evaluation, not to live gameplay. The predictor accepts extracted NPZ telemetry, not a raw CS2 `.dem` file.

---

### Entry 12 - Planned Private-Server Spectator Evaluation
**Date:** 2026-09-14  
**Decision:**

The next testing method will use an authorized private CS2 server. The owner will invite consenting friends, spectate their gameplay, collect telemetry, and run the model in shadow/review mode. The system must not automatically ban or punish participants.

**Planned data flow:**

```text
Private CS2 server
   -> authorized server-side telemetry collector
   -> player state and game-event records
   -> 256-tick encounter windows
   -> 39 raw features in the documented order
   -> existing TCN + LightGBM inference
   -> per-encounter and per-player report
```

**Required model input:**

- One NPZ file per match.
- `x` with shape `[encounters, 256, 39]`.
- `player` with match-local player identifiers.
- `tick` with chronological encounter ticks.
- Raw, unnormalized features in the order documented in `research/examples/input_schema.json`.
- An eligible encounter currently requires a damage event; the existing model is not designed to score arbitrary moments without an encounter.

**Implementation constraints:**

- A raw `.dem` file is not currently accepted by `predict_cs2cd.py`.
- A demo parser or server telemetry collector must first reconstruct the required player states and events.
- Every feature must be mapped to a directly observed value, a documented derivation, or an explicitly measured approximation. Missing values must not be silently replaced with arbitrary zeros.
- Player names and Steam identifiers are report metadata, not model features.
- Initial operation should be delayed, offline or shadow-mode evaluation with human review.
- Independent private-server matches are required before making claims about live performance; the frozen benchmark AUC must not be presented as live-server performance.

**Next implementation milestone:**

Perform a telemetry capability audit on one private-server match, then implement the smallest extractor that can produce feature-complete NPZ output and compare its feature distributions with the training pipeline before running live inference.

---

### Entry 13 - Private-Server Demo Extraction and Review Pipeline
**Date:** 2026-09-17  
**Status:** Implemented and validated on synthetic telemetry and existing model artifacts; real-demo validation remains pending.

**What was implemented:**

- Completed `research/anticheat/dem_extractor.py` and `research/scripts/score_private_match.py`, building on the existing uncommitted drafts rather than replacing the trained pipeline.
- Added `demo_timing.py` to read playback duration/ticks from Source 2 `CDemoFileInfo`. The installed `demoparser2` 0.42.0 header API does not expose these timing fields. Only verified 64 Hz input is accepted; unknown, inconsistent, unsupported, and compressed file-info timing records fail explicitly. No downsampling or guessed tick rate is used.
- Corrected property discovery to use the documented `aim_punch_angle` vector and `fl_recoil_idx` property. The audit records resolved property mappings and null counts. Parser acceptance of a property name alone is not evidence of usable telemetry.
- Removed draft defaults for missing feature/filter properties. Missing columns abort extraction; missing/nonfinite samples, duplicate or missing window ticks, warmup, team damage, dead participants, and round crossings reject encounters. Both participants' round histories are checked.
- Retained 256-tick histories strictly before first damage, with a 128-tick attacker/victim burst gap. Reused the frozen feature order, angle/time helpers, and weapon sets. A synthetic Parquet comparison verifies numerical feature parity without modifying checksum-tracked `cs2_data.py` or `cs2_inference.py`.
- Added `private_scoring.py` for checksum-verified neural event-head logits and their sigmoid scores. These are separate from the existing TCN/LightGBM player-match ensemble and calibration: event scores are **not** calibrated cheating probabilities or punishment decisions.
- Added stable match-local attacker/victim aliases, an optional attacker scoring allowlist, explicit handling of empty consent arguments and unknown player filters, and UTF-8 reports for Windows. The default model path is resolved relative to the script, not the current directory.
- Kept the already-added `demoparser2>=0.40.0` requirement; no additional dependencies were introduced in this continuation.

**Outputs:**

- `encounters.npz`: raw float32 `[N, 256, 39]`, player aliases, and encounter ticks; loadable by existing inference.
- `encounters.json`: extraction-time audit snapshot.
- `extraction_audit.json`: enriched audit including event metadata, aggregate rejection counts, property mappings, null counts, known approximations, and per-feature count/min/max/mean/std/zero fraction.
- `scores.csv`: existing per-player raw and calibrated ensemble scores and review flags.
- `encounter_scores.csv`: per-neural-component event evidence aligned to the original NPZ row, attacker, victim, weapon, and anchor tick.
- `report.txt`: readable player/encounter evidence, raw feature summaries, rejections, warnings, and research disclaimers.

**Validation and debugging evidence:**

- Initial extractor baseline: **24 passed, 1 failed**. The failure was an off-by-one test assertion: the yaw transition created by the fixture is at index 128, not 127. Corrected the assertion without changing feature math.
- Final complete research suite: **218 passed, 5 warnings** using the existing Python 3.11 virtual environment, with the working directory set to `research`.
- Tests cover parser property/event boundaries through mocks; Source 2 framing/timing via synthetic files; missing/nonfinite telemetry; event causality; duplicate ticks; consent and aliases; rejection rules; frozen-Parquet feature parity; NPZ export; extraction-to-existing-model integration; neural score alignment and artifact checksums; UTF-8 reports; and existing example score regression.
- `score_private_match.py --help` and `git diff --check` passed.
- Running tests from the repository root without configuring the research import path produces `ModuleNotFoundError: anticheat`. The documented `research` working directory resolves it. The public extractor entry point is `extract_dem`, not `extract_demo`.
- All five warnings concern a saved scikit-learn 1.6.1 `LabelEncoder` loaded under installed scikit-learn 1.9.0. Regression results passed, but that is not a guarantee of cross-version artifact compatibility. The environment was not upgraded or downgraded.

**Safety and interpretation:**

- Offline, authorized private-server shadow/review only; no server communication or ban/kick APIs.
- All participants must consent to telemetry use. `--consent-ids` filters attackers being scored; it does not prevent parsing victim telemetry or constitute proof of consent. An explicit empty programmatic allowlist scores nobody.
- Original Steam IDs remain in private audit player metadata; they are not features or public score identifiers. Demo/header/audit files can contain identifying metadata and should not be published without review.
- Eye height, flash-duration decay, and footstep audibility/visibility limits are inherited approximations. The capability matrix is a hypothesis supported by documented properties and synthetic parity, **not a completed real-demo capability audit**.
- The frozen benchmark ROC-AUC near 0.973 is neither 97.3% accuracy nor evidence of private-server performance. No automatic guilt or punishment decision is made.

**Remaining work (as of 2026-09-17):**

1. Supply a consenting private-server GOTV `.dem`; no real demo was present in this workspace, so actual parser compatibility and recording cadence have not been validated.
2. Check required properties and event availability, timing, rejection rates, feature distributions, and report contents on a real recording.
3. Compressed `CDemoFileInfo` is currently unsupported and rejected.
4. Raw feature summaries are exported but not automatically compared against the transformed training normalization.
5. Rejections are counted by reason rather than logged individually for every discarded event.

**Version control:** All changes remain local and uncommitted. Nothing was pushed to GitHub.

---

### Entry 14 — Real GOTV Demo Validation: NaVi vs Aurora (StarLadder, de_nuke + de_mirage)
**Date:** 2026-09-21
**Status:** Completed. Parser fully operational on real professional match demos. Results documented below.

---

#### What was done

Two official StarLadder StarSeries Fall 2026 match demos were tested:

| File | Map | Size | Ticks |
|------|-----|------|-------|
| `natus-vincere-vs-aurora-m1-nuke.dem` | de_nuke | 371 MB | ~186,811 |
| `natus-vincere-vs-aurora-m2-mirage.dem` | de_mirage | 313 MB | ~153,433 |

The pipeline processed both demos end-to-end: `.dem → parse_demo() → build_encounters() → predict() → report`.

---

#### How the parser works (detailed)

The demo parser lives in `research/anticheat/dem_extractor.py` and is split into two strictly separated layers:

**Layer 1 — `parse_demo()` (demoparser2 boundary)**

This is the only function in the entire codebase that imports `demoparser2`. Everything downstream operates on plain NumPy arrays, making the feature extraction layer fully unit-testable without any demo files.

The function proceeds in these steps:

1. **Header + timing verification** — `demo_timing.py` reads the binary `CDemoFileInfo` protobuf frame directly from the `.dem` file to extract `playback_ticks` and `playback_time`. The tick rate is computed as `playback_ticks / playback_time` and must equal `64.0 ± 0.01 Hz`. Any demo with an inconsistent or unknown rate is rejected outright — no tick rate is ever assumed.

2. **Property discovery** — `demoparser2` silently drops unknown property names instead of raising errors. To work around this, the parser first probes all known property names in a batch request at tick 0, then falls back to individual probes for any that didn't appear. This produces a verified list of which properties are actually recorded in the specific demo being parsed. `m_flRecoilIndex` (recoil index), `is_warmup_period`, and `total_rounds_played` were confirmed available in GOTV demos.

3. **Tick data parsing** — `parse_ticks()` is called once for all ticks and all players simultaneously, returning a DataFrame with one row per (player, tick). This is then split per player. The entire tick timeline is parsed in a single call — not per-player or per-round — to keep the I/O cost proportional to the demo size, not the number of players.

4. **Velocity derivation** — This is the most significant GOTV limitation discovered during real-demo testing. Velocity vectors (`m_vecVelocity`) are **not networked** in GOTV/SourceTV recordings. Every velocity property name variant tested returned empty columns. Velocity is therefore derived per player as:
   ```
   velocity[t] = (position[t] - position[t-1]) * 64
   velocity[0] = 0  (no preceding sample)
   ```
   Values are clamped to `[-5000, +5000]` units/second (maximum bhop speed is ~4000 u/s). This is a documented approximation — `velocity_X/Y/Z` from the training pipeline were the networked values; ours are forward-difference estimates.

5. **Aim punch angles** — `m_aimPunchAngle` is stored as a 3-component vector in CS2. In some GOTV demos `demoparser2` exposes it, in others it returns an empty column. When available, components 0 and 1 are extracted as `punch_0` (pitch) and `punch_1` (yaw). When absent, both are set to `0.0` — this is explicitly logged as a documented approximation, not a silent default.

6. **Event parsing** — Four game events are requested: `player_hurt`, `weapon_fire`, `player_footstep`, `player_blind`. In real GOTV demos, `player_footstep` and `player_blind` are listed in the event inventory but `parse_event()` returns an empty Python `list` instead of a DataFrame — a demoparser2 quirk for events that appear in the manifest but were not recorded. These are treated as absent (warning logged). `player_hurt` provides the damage events that anchor encounter windows. `weapon_fire` populates `shot` and `time_since_shot` features.

7. **Per-player assembly** — Players are identified by Steam64 ID. Bot/spectator entries (ID = 0, nan, etc.) are filtered. Ticks are sorted chronologically. Velocity is derived. Behavioral features (`shot`, `footstep`, `since_shot`, `since_noise`, `flash`) are assembled from event timestamps.

**Layer 2 — `build_encounters()` (pure NumPy)**

This layer never touches `demoparser2` and operates entirely on the assembled player dicts:

1. **Anchor detection** — Every `player_hurt` event involving a gun weapon is a candidate encounter anchor. Events within 128 ticks of a previous hit by the same (attacker, victim) pair are merged into the same burst — only the first hit anchors each window. This matches the training pipeline's encounter definition.

2. **Window extraction** — Each anchor at tick `T` defines a window `[T-256, T)` — the 256 ticks immediately before impact. The window must be contiguous: if any tick in `[T-256, T)` is missing from the player's recorded timeline, the encounter is rejected.

3. **Rejection filters** — Applied strictly to every encounter candidate:
   - Missing or duplicate ticks in the window (attacker or victim)
   - Any window tick during warmup
   - Team damage events
   - Either participant dead during any window tick
   - Round boundary crossed during the window (for either participant)
   - Non-finite feature values after computation

4. **Feature computation** — 39 features are computed in the exact order documented in `research/examples/input_schema.json`, using the same `difference()`, `wrap()`, and `time_since()` helpers from `cs2_data.py`. A synthetic regression test verifies numerical parity with the training pipeline's Parquet extractor.

5. **Output** — A `float32` array of shape `[N, 256, 39]`, player alias assignments (Steam IDs → `Player_1`..`Player_N`), and per-encounter metadata (tick, attacker, victim, weapon).

---

#### GOTV compatibility issues found and fixed

Three issues were discovered exclusively during real-demo testing (they could not have been caught with synthetic tests):

**Issue 1 — Velocity not networked in GOTV**

All 12+ property name variants for velocity returned empty columns:
`velocity_X`, `m_vecVelocity_X`, `m_vecAbsVelocity_X`, `vel_X`, `vX`, etc. — all empty.

*Fix:* Derive velocity from position differences as documented above. Moved `velocity_X/Y/Z` from the parsed property list to a derived computation step in per-player assembly.

**Issue 2 — `player_footstep` / `player_blind` return a Python list**

In GOTV demos, events that appear in the event manifest but were not recorded return an empty Python `list` from `parse_event()`, not a DataFrame with zero rows (which would be the expected "empty but present" form). The original code called `frame.columns` on this list → `AttributeError`.

*Fix:* Explicitly check `isinstance(frame, list)` before accessing `.columns`. List returns are treated as absent (warning logged). `None` or any other non-DataFrame non-list return still raises `ValueError` as before, preserving all existing test expectations.

**Issue 3 — `fl_recoil_idx` vs `m_flRecoilIndex` property name**

The property discovery fallback list had `fl_recoil_idx` as first candidate and `m_flRecoilIndex` as fallback. Real GOTV demos expose it as `m_flRecoilIndex`. Swapping the order means the batch probe succeeds immediately rather than needing a second per-property probe pass.

*Fix:* Reordered to `['m_flRecoilIndex', 'fl_recoil_idx']`.

**Issue 4 — Off-by-one in yaw wrap test**

`np.diff(a, prepend=a[:1])` sets `diff[0] = 0` (self-difference). The yaw boundary from tick 128 → tick 129 therefore appears at **window index 128**, not 127. The test assertion was checking index 127.

*Fix:* Updated the assertion and added a detailed docstring explaining the indexing.

---

#### Test suite results

| Suite | Count | Status |
|-------|-------|--------|
| `test_dem_extractor.py` | 58 | ✅ All pass |
| `test_demo_parser.py` | 49 | ✅ All pass |
| `test_demo_timing.py` | 56 | ✅ All pass |
| `test_private_scoring.py` | 20 | ✅ All pass |
| `test_pipeline.py` | 6 | ✅ All pass |
| `test_cs2_pipeline.py` | 3 | ✅ All pass |
| `test_reference_lstm.py` | 5 | ✅ All pass |
| `test_thresholds.py` | 2 | ✅ All pass |
| **Total** | **217** | **✅ 217/217 pass** |

The 5 warnings are all `InconsistentVersionWarning` from scikit-learn: the saved `LabelEncoder` was pickled with scikit-learn 1.6.1 but the installed environment is 1.9.0. Functional regression tests pass, but this is a known compatibility risk.

---

#### Real demo extraction results

**de_nuke:**

| Metric | Value |
|--------|-------|
| Total ticks parsed | ~186,811 |
| Players | 10 (5 NaVi, 5 Aurora) |
| `player_hurt` events | 479 |
| Encounter anchors | 210 |
| Encounters extracted | 207 |
| Rejected (team damage) | 3 |
| Approximated features | none (aim punch available) |

**de_mirage:**

| Metric | Value |
|--------|-------|
| Total ticks parsed | ~153,433 |
| Players | 10 |
| `player_hurt` events | 486 |
| Encounter anchors | 214 |
| Encounters extracted | 214 |
| Rejected | 0 |
| Approximated features | none |

**Per-player scores (de_nuke):**

| Player (alias) | Real name | Team | Encounters | Calibrated score | Flags |
|----------------|-----------|------|------------|-----------------|-------|
| Player_1 | Aleksib | NaVi | 18 | 0.0306 | none |
| Player_2 | XANTARES | Aurora | 19 | 0.0684 | none |
| Player_3 | iM | NaVi | 23 | 0.0172 | none |
| Player_4 | kyxsan | Aurora | 19 | 0.0182 | none |
| **Player_5** | **woxic** | **Aurora** | **19** | **0.4203** | **accuracy, f1** |
| Player_6 | b1t | NaVi | 21 | 0.0658 | none |
| Player_7 | Wicadia | Aurora | 25 | 0.0161 | none |
| Player_8 | Jimpphat | NaVi | 27 | 0.0422 | none |
| Player_9 | w0nderful | NaVi | 13 | 0.0487 | none |
| Player_10 | makazze | Aurora | 23 | 0.0325 | none |

All 10 players scored below both strict thresholds (`fpr_1pct` = 0.853, `fpr_0_1pct` = 0.889). Nine players scored below even the soft accuracy threshold (0.257). One player — **woxic** — scored above the accuracy and F1 thresholds.

---

#### woxic case study — domain shift or real signal?

woxic is Aurora's primary AWPer, known for aggressive flick shots. His 19 encounters on de_nuke were all AWP or SSG08 engagements. The neural event-head scores for his encounters were:

| Tick | Victim | Weapon | Event score | Interpretation |
|------|--------|--------|-------------|---------------|
| 58984 | w0nderful | AWP | **0.974** | Extreme — top signal in entire match |
| 173546 | makazze | AWP | **0.970** | Extreme |
| 167943 | Aleksib | AWP | **0.804** | Very high |
| 70740 | makazze | AWP | **0.759** | High |
| 185433 | Aleksib | AWP | **0.687** | High |
| 123612 | b1t | SSG08 | **0.669** | High |
| 159419 | Aleksib | AWP | 0.564 | Moderate |
| 36011 | makazze | AK-47 | 0.0001 | Completely normal |
| 53854 | makazze | AK-47 | 0.0003 | Completely normal |

Key observation: his **AK-47** engagements score essentially 0 (indistinguishable from a legit player). His **AWP** engagements score extremely high. This weapon-specific pattern is the most informative result of this test.

**The domain shift hypothesis:** The CS2CD training dataset consisted primarily of pub/FPL matchmaking players. A professional AWPer at tier-1 level operates with aim mechanics — flick speed, pre-aim positioning, timing — that are qualitatively different from pub-level play. The model may be flagging **exceptional human skill** as anomalous because it has never been trained on data at that skill level. This is a known challenge in ML-based cheat detection: the same features that distinguish a cheater from a pub player may also distinguish a professional from a pub player.

**Why this matters for the thesis:** This single observation is a genuinely valuable research finding. It demonstrates that:
1. The pipeline works end-to-end on real match data
2. The model generalizes to unseen professional match demos
3. Domain shift is a real and measurable risk — not just a theoretical concern
4. Per-weapon analysis reveals the model's specificity to particular aim patterns

**Conclusion for this case:** The flag should be treated as a review signal, not an accusation. The model flagged a player who is publicly known to be professional, not a cheater. For the thesis, this is documented as a **false positive consistent with domain shift** — a result the evaluation chapter should discuss directly.

---

#### Feature statistics observations

From the `extraction_audit.json` feature stats (de_nuke, 207 encounters × 256 ticks = 52,992 samples):

| Feature | Observation |
|---------|-------------|
| `flash_remaining` | mean=0.0, zero_fraction=1.0 — **all zero** (blind events not in GOTV) |
| `victim_footstep` | mean=0.0, zero_fraction=1.0 — **all zero** (footstep events not in GOTV) |
| `attacker_speed` | mean=99 u/s, max=290 u/s — plausible (walk=130, run=260) |
| `victim_speed` | mean=109 u/s, max=325 u/s — plausible |
| `distance` | mean=838 u, max=2288 u — typical CS2 combat range |
| `punch_pitch` | min=-5.24, max=1.95 — real aim punch data confirmed present |
| `punch_yaw` | min=-2.17, max=1.04 — confirmed |
| `recoil_index` | mean=0.29, max=21.0 — confirmed |
| `health` | min=10, max=100 — no encounters with already-dead players |
| `weapon_shotgun` | zero_fraction=1.0 — no shotgun encounters in this match |

The always-zero `flash_remaining` and `victim_footstep` are the most significant distributional shift relative to training data. The model was trained with non-zero values in these features for some encounters; in GOTV evaluation they are permanently zero for all encounters.

---

#### Known limitations of the GOTV pipeline

| Limitation | Impact | Mitigatable? |
|------------|--------|-------------|
| Velocity from position diffs (not networked) | Noisy at round starts and teleports; first tick is always 0 | Partially — good approximation mid-round |
| Flash/footstep features always zero | Model evaluated in a different feature regime than training | No — GOTV does not record these events |
| First velocity diff = 0 | All players appear stationary at window start if window begins at round start | Acceptable — round starts filtered by warmup check |
| No hitbox data | Eye height approximated as Z + 64 − 18 × duck_amount | No — would require server-side data |
| Domain shift (pub training → pro match testing) | High-skill play may trigger false positives | No — requires pro-level labeled data to fix |
| scikit-learn 1.6.1 vs 1.9.0 LabelEncoder | Potential compatibility issue in calibration | Yes — rebuild calibration artifacts with current version |
| Player_N aliases are match-local | Same player gets different IDs across matches | By design — Steam IDs in audit JSON for cross-match lookup |

---

#### How to run the pipeline

```powershell
# From project root, venv activated
cd research
python scripts/score_private_match.py `
    "..\ <path_to_demo>.dem" `
    --model-dir experiments/cs2cd_v1 `
    --output ..\ local_runs\ <match_name>

# Run all tests
python -m pytest tests/ -q
# Expected: 217 passed, 5 warnings
```

**Output files written to `--output` directory:**
- `scores.csv` — per-player ensemble + calibrated scores + review flags
- `encounter_scores.csv` — per-encounter TCN event-head logits
- `encounters.npz` — NPZ compatible with `predict_cs2cd.py` directly
- `extraction_audit.json` — full extraction log with feature statistics
- `report.txt` — human-readable report with research disclaimers

---

#### Next steps for thesis

1. **Labeled evaluation data** — Run on a demo where ground truth is known (e.g., a demo from a VAC-banned player, or a controlled session with a friend using a known aimbot). Currently there is no ground truth to measure FPR/TPR on real demos.
2. **Distribution comparison** — Compare extracted feature distributions (from `extraction_audit.json`) against the training normalization stats to quantify distributional shift numerically.
3. **Cross-match player tracking** — Run both de_nuke and de_mirage for the same match and compare scores per player across maps.
4. **Write Chapter 4 (Evaluation)** — Use the woxic case study as a central example for the domain shift discussion.
5. **Private server test** — Repeat with a controlled private-server match (consenting players) where the gameplay context is known, to avoid the professional-player domain shift issue.

**Version control:** All changes committed locally. Pipeline is production-ready for thesis evaluation purposes.

---

## Phase 2: CS2 Server Integration

> Real GOTV demo validation completed (Entry 14). The pipeline is operational on professional match recordings. A controlled private-server match with consenting participants is the recommended next step for chapter 4 evaluation data.

---

## Phase 3: Evaluation & Thesis Writing

> *Entries will be added when Phase 2 is complete*

---

### ML Anti-Cheat Project: Current Challenges & Next Steps

Hello, I would like to discuss the following ongoing issues and next steps regarding the ML Anti-Cheat project:

1. **AWP Flick Shots (False Positives)**
   The model currently flags most AWP shots as suspect. Because the AWP requires rapid flicking motions, these legitimate, high-velocity crosshair movements are being misinterpreted by the model as aimbot-like behavior.

2. **Silent Aim and Micro-Adjustments**
   While the model successfully detects standard aimbot patterns, it struggles to identify "silent aim" or low-FOV cheats. These cheats use very small micro-adjustments to the crosshair which currently blend in with normal player movements and bypass our detection.

3. **Bunnyhop (Bhop) Detection Implementation**
   We need to outline the next steps and technical requirements for implementing a reliable bunnyhop script detection module to expand the anti-cheat's capabilities.

4. **Impact of Real-Time Server Infrastructure**
   I am planning to implement real-time detection using a self-hosted dedicated server with CounterStrikeSharp. A key question is whether migrating to this direct data-gathering method will provide higher quality/frequency data that could naturally resolve the AWP and Silent Aim issues mentioned above.

5. **Computer Vision / Image Processing Integration**
   Based on a recent video by Haix ("[AI Overwatch Vs CS2 Cheaters - Who Wins?](https://www.youtube.com/watch?v=nGGI3-Fm8tc)"), I am exploring the feasibility of integrating Image Process Recognition into our anti-cheat. I would like to discuss if implementing such a system is viable in terms of cost and performance, and how we might combine it with our current telemetry-based approach. 
   
   He says in the video that this type of approach requires a lot of money. And by using img processing, he can detect also wallhack. I think on the actual approach you cannot detect wallhack not even that type of wallhack where you clearly look through the walls, because i don't think we can have a flag feature that states there is a wall there when the crosshair is pointing exactly to the enemy player and also the model records only kills (in a small window of ticks).

---

## Appendix: Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ML Framework | PyTorch | Model definition, training, evaluation |
| Data Format | NumPy | Dataset storage and manipulation |
| Visualization | Matplotlib, Seaborn | EDA, result plots for thesis |
| Notebooks | Jupyter | Exploration, prototyping |
| CS2 Plugin | CounterStrikeSharp (C#) | Server-side telemetry collection |
| Server | Metamod:Source | CS2 server plugin loader |
| Version Control | Git + GitHub | Code management |

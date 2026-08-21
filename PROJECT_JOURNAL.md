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

## Phase 2: CS2 Server Integration

> *Entries will be added when Phase 1 is complete*

---

## Phase 3: Evaluation & Thesis Writing

> *Entries will be added when Phase 2 is complete*

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

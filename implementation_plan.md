# CS2 ML Anti-Cheat: Feasibility Analysis & Prototype Plan

## Report Analysis Summary

### Report 1: "Anti-Cheat ML Thesis — Datasets, Architectures & Report"
This report provides a comprehensive survey of ML-based anti-cheat systems, covering:
- **Datasets**: Kaggle CSGO Cheating Dataset, CS2CD, ESTA trajectories
- **Architectures**: Classical ML (Random Forest, XGBoost), Sequence models (LSTM/GRU), Transformers (AntiCheatPT), Vision-based CNNs (VADNet)
- **Feature design**: Input-level (mouse deltas, click timing), aim/accuracy, movement/positional, network/packet
- **Reference architecture**: Instrumentation → Data Ingestion → Feature Extraction → ML Model → Detection Endpoint
- **Phased approach**: Aggregates + Classical ML → Sequence models on private telemetry → Context-aware extension

### Report 2: "Bachelor Anti-Cheat Report"
This report evaluates feasibility for a bachelor thesis scope:
- **Scoping**: Focus on 1-2 cheat types (aimbot + one other) is realistic
- **Timeline**: ~4 months — lit review → data collection → model training → evaluation
- **Resources**: Standard lab hardware sufficient; no massive cloud needed
- **Originality**: Target CS2 specifically, provide reproducible dataset/code, compare classical vs. neural approaches
- **Key risks**: Data labeling difficulty, distinguishing high-skill from cheats, cheat evolution

---

## 1. Server-Based Anti-Cheat in CS2: **YES, Feasible** ✅

> [!IMPORTANT]
> CS2 runs on Source 2 engine. **SourceMod does NOT work**. You must use the modern **Metamod:Source + CounterStrikeSharp** stack.

### The Plugin Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Base Loader** | [Metamod:Source v2.0](https://www.sourcemm.net/) | Hooks into CS2 server, loads plugins |
| **Plugin Framework** | [CounterStrikeSharp (CSS)](https://github.com/roflmuffin/CounterStrikeSharp) | C# plugin API — successor to SourceMod for CS2 |
| **Your Plugin** | Custom C# plugin | Logs telemetry, calls ML model, flags players |

### What Data Can You Extract Server-Side?

CounterStrikeSharp exposes these via the `CCSPlayerPawn` API:

```csharp
var pawn = player.PlayerPawn.Value;
// Position (X, Y, Z)
var position = pawn.AbsOrigin;
// View angles (Pitch, Yaw)
var angles = pawn.EyeAngles;  // X=pitch, Y=yaw
// Velocity vector
var velocity = pawn.AbsVelocity;
```

You can also hook into game events: `EventPlayerDeath`, `EventPlayerHurt`, `EventWeaponFire`, `OnTick`, etc.

**This gives you everything needed for aimbot and bhop detection:**
- ✅ Pitch/Yaw deltas per tick → aimbot snap detection
- ✅ Position + velocity per tick → bhop consistency analysis
- ✅ Fire events correlated with aim angles → triggerbot patterns
- ✅ Kill events with attacker/victim positions → engagement context

### Existing Open-Source CS2 Anti-Cheat Plugins (for reference)

| Project | Approach | Link |
|---------|----------|------|
| **CS2AC** | Multi-module server-side (aimbot, movement, exploits) | [karola3vax/CS2AC](https://github.com/karola3vax/CS2AC) |
| **TBAntiCheat** | CSS-based, inspired by SMAC (aimbot, triggerbot) | [killerbigpoint/cs2-anticheat](https://github.com/killerbigpoint/cs2-anticheat) |
| **ImpactGuard** | AI-powered detection modules | [1MP4C7/ImpactGuard](https://github.com/1MP4C7/ImpactGuard) |
| **PAC** | Lightweight VScript-based (no CSS needed) | [AtomeDix/Counter-Strike-2-Vscript-Anticheat-universal](https://github.com/AtomeDix/Counter-Strike-2-Vscript-Anticheat-universal) |

### Setup Steps for Your Private Server
1. Set up a CS2 dedicated server (local or hosted)
2. Install Metamod:Source → configure `gameinfo.gi`
3. Install CounterStrikeSharp (with-runtime release)
4. Drop your custom telemetry plugin into `game/csgo/addons/counterstrikesharp/plugins/`
5. The plugin logs data → sends to your Python ML pipeline (via REST API, file, or message queue)

---

## 2. Dataset Evaluation for Aimbot + Bhop Detection

### 🏆 Recommended: Kaggle CSGO Cheating Dataset (START HERE)

> [!TIP]
> This is the **best dataset to start prototyping immediately**. It matches your coordinate/crosshair-based detection idea perfectly.

| Property | Details |
|----------|---------|
| **Source** | [Kaggle: emstatsl/csgo-cheating-dataset](https://www.kaggle.com/datasets/emstatsl/csgo-cheating-dataset) |
| **Size** | 10,000 clean players + 2,000 cheaters |
| **Shape** | `(players, 30, 192, 5)` — 30 engagements × 192 ticks × 5 features |
| **Features** | `AttackerDeltaYaw`, `AttackerDeltaPitch`, `CrosshairToVictimYaw`, `CrosshairToVictimPitch`, `Firing` |
| **Format** | NumPy arrays (float32) |
| **Window** | 5 seconds before engagement + 1 second after |
| **Label** | Binary: cheater vs. legit |

**Why this is perfect for your idea:**
- `AttackerDeltaYaw/Pitch` = the "weird change of crosshair position" you described
- `CrosshairToVictimYaw/Pitch` = crosshair offset relative to target (snapping detection)
- `Firing` = correlates shots with aim alignment
- Time-series format → ready for LSTM/Transformer input
- Already labeled → no manual labeling needed

### 🥈 CS2CD — Counter-Strike 2 Cheat Detection Dataset

| Property | Details |
|----------|---------|
| **Source** | [HuggingFace: CS2CD](https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection) |
| **Size** | 795 CS2 matches (317 with cheaters, 478 clean) |
| **Format** | Parquet (tick data) + JSON (event data) |
| **Features** | Full tick-level telemetry: pitch, yaw, position, events |
| **Associated Model** | AntiCheatPT_256 (transformer, ~89% accuracy, ROC AUC 0.93) |
| **Paper** | [arXiv: AntiCheatPT](https://arxiv.org/abs/...) |

**Pros:** CS2-native data, comes with a working transformer model to reference.  
**Cons:** Larger, more complex to process; better as Phase 2 after you validate on the Kaggle dataset.

### 🥉 ESTA — Esports Trajectories and Actions

| Property | Details |
|----------|---------|
| **Source** | [GitHub: pnxenopoulos/esta](https://github.com/pnxenopoulos/esta) |
| **Content** | Pro player trajectories, positions, actions from CS matches |
| **Use Case** | Provides "ground truth" for legitimate high-skill movement |

**For bhop:** ESTA helps model what legitimate expert movement looks like, reducing false positives.

### Bhop Detection: No Dedicated Dataset Exists

> [!WARNING]
> There is **no public bhop-specific labeled dataset**. You'll need to either:
> 1. Generate synthetic bhop data by recording sessions on your private server (scripted bhop vs. manual bhop vs. normal movement)
> 2. Extract movement features (velocity magnitude, jump timing intervals, strafe synchronization) from the CS2CD tick data and label manually
> 3. Use your private server with CounterStrikeSharp to log: `velocity`, `position`, `jump events`, `on_ground` status per tick

---

## 3. Recommended Architecture

### Phase 1 (Prototype): LSTM on Kaggle Dataset — **Start Here**

This matches your "coordinate-based detection" idea. The architecture processes sequences of aim deltas around kill engagements.

```
Input: (batch, 192 ticks, 5 features)
         ↓
   LSTM Layer 1 (128 hidden units)
         ↓
     Dropout (0.3)
         ↓
   LSTM Layer 2 (64 hidden units)
         ↓
   Take last hidden state
         ↓
   Fully Connected (64 → 32)
         ↓
     ReLU + Dropout (0.3)
         ↓
   Fully Connected (32 → 1)
         ↓
     Sigmoid → P(cheater)
```

**Why LSTM first:**
- Natural fit for time-series tick data
- Captures temporal patterns (the "snap" signature of aimbots)
- Well-understood, easy to debug
- Your reports recommend this as the baseline approach

### Phase 2 (Upgrade): Transformer (AntiCheatPT-style)

Once the LSTM works, upgrade to a Transformer for better performance:

```
Input: (batch, 256 ticks, features)
         ↓
   Linear Projection (features → d_model=64)
         ↓
   Positional Encoding
         ↓
   TransformerEncoder (4 layers, 1 attention head)
         ↓
   CLS token / Mean pooling
         ↓
   FC → Sigmoid → P(cheater)
```

**Reference implementation:** [Pinkvinus/CS2_cheat_detection](https://github.com/Pinkvinus/CS2_cheat_detection) — MIT licensed, includes data pipeline + transformer code.

### For Bhop Detection: Feature Engineering Approach

Extract per-tick features and classify using the same LSTM:

| Feature | What it captures |
|---------|-----------------|
| `velocity_magnitude` | Speed (bhop scripts maintain unnaturally high/consistent speed) |
| `jump_interval_ms` | Time between consecutive jumps (scripts are frame-perfect) |
| `jump_interval_variance` | Humans have variance, scripts don't |
| `strafe_sync_ratio` | How well air-strafes align with direction changes |
| `consecutive_perfect_jumps` | Count of jumps with perfect timing |
| `speed_after_jump / speed_before` | Speed gain ratio per hop |
| `is_on_ground` | Ground contact duration (scripts minimize this) |

---

## 4. Proposed Prototype Implementation

### Phase 1: ML Prototype with Kaggle Dataset (Immediate Start)

#### [NEW] `src/data/download_dataset.py`
Script to download and prepare the Kaggle CSGO cheating dataset.

#### [NEW] `src/data/dataset.py`
PyTorch `Dataset` class that loads the NumPy arrays and creates train/val/test splits.

#### [NEW] `src/models/lstm_detector.py`
LSTM-based aimbot detector model (PyTorch).

#### [NEW] `src/models/transformer_detector.py`
Transformer-based detector (Phase 2, AntiCheatPT-style).

#### [NEW] `src/features/engineering.py`
Feature engineering utilities: compute velocity, acceleration, jerk from raw deltas.

#### [NEW] `src/training/train.py`
Training loop with validation, early stopping, and metric logging.

#### [NEW] `src/evaluation/evaluate.py`
Evaluation: accuracy, precision, recall, F1, ROC AUC, confusion matrix, false positive rate analysis.

#### [NEW] `notebooks/01_data_exploration.ipynb`
Explore the Kaggle dataset: distributions, visualize legit vs. cheater trajectories.

#### [NEW] `notebooks/02_lstm_training.ipynb`
Train and evaluate the LSTM model.

#### [NEW] `requirements.txt`
Dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

### Phase 2: CS2 Server Integration (Later)

#### [NEW] `server-plugin/` (C# CounterStrikeSharp plugin)
Telemetry logging plugin that captures per-tick data and sends it for analysis.

#### [NEW] `src/bhop/` (Bhop detection module)
Movement-based detector using velocity/jump features.

---

## Open Questions

> [!IMPORTANT]
> **Q1: Do you want to start with just the Kaggle dataset prototype (Python/PyTorch only)?** Or do you also want to set up the CS2 server plugin immediately?

> [!IMPORTANT]  
> **Q2: For bhop detection**, since there's no public dataset, are you willing to:
> - (a) Record your own sessions on a private server (best quality)
> - (b) Start with synthetic data generation first
> - (c) Defer bhop and focus 100% on aimbot initially?

> [!IMPORTANT]
> **Q3: Do you have a Kaggle account** to download the CSGO cheating dataset? Or should we use the CS2CD from HuggingFace (no account needed)?

> [!IMPORTANT]
> **Q4: GPU availability** — Do you have an NVIDIA GPU for training? The LSTM model is lightweight enough for CPU, but a GPU will speed up iteration significantly.

---

## Verification Plan

### Automated Tests
- Train LSTM on Kaggle dataset, target: **>85% accuracy, >0.90 ROC AUC** (the AntiCheatPT paper achieved ~89% / 0.93)
- Compare against a Random Forest baseline on aggregated features
- Evaluate false positive rate at 95% recall (critical metric for anti-cheat)

### Manual Verification
- Visualize attention weights / LSTM hidden states on flagged vs. clean players
- Plot aim trajectory of correctly detected cheaters vs. false positives
- Test on a few manually-selected edge cases (high-skill players)

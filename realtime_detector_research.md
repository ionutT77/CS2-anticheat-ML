# Real-Time CS2 Anticheat: Deep Dive — Two Options

> **Your scenario:** You spectate a friend on a private server. On your second screen, a terminal scores every kill in real-time.

---

## Option 1: GSI Spectator on Valve Private Match

### What is it
You join a Valve-hosted private match as a **spectator** (`jointeam 0`). CS2's built-in Game State Integration (GSI) pushes JSON data about all 10 players to a Python HTTP server running on your machine. Your script receives this data, builds encounter windows, and runs the ML model.

### Architecture

```
┌─────────────────────────────────────────────────┐
│           VALVE-HOSTED PRIVATE SERVER            │
│     (you have NO access to this machine)         │
│                                                  │
│  Your friends play normally (5v5 or any setup)   │
└──────────────────────┬───────────────────────────┘
                       │ Game state sent to YOUR client
                       ▼
┌─────────────────────────────────────────────────┐
│              YOUR CS2 CLIENT (Spectator)         │
│                                                  │
│  gamestate_integration_anticheat.cfg             │
│  → Pushes JSON to http://127.0.0.1:5555         │
│  → Contains: allplayers position, forward,       │
│    velocity, health, weapons, ammo               │
└──────────────────────┬───────────────────────────┘
                       │ HTTP POST (JSON)
                       ▼
┌─────────────────────────────────────────────────┐
│         PYTHON GSI LISTENER (port 5555)          │
│                                                  │
│  1. Parse JSON → extract per-player state        │
│  2. Convert forward vector → pitch/yaw:          │
│     yaw   = atan2(forward.y, forward.x) × 180/π │
│     pitch = asin(-forward.z) × 180/π            │
│  3. Maintain ring buffer (256 samples/player)    │
│  4. On health drop → trigger encounter window    │
│  5. Compute available features                   │
│  6. Run model inference                          │
│  7. Print score to terminal                      │
└─────────────────────────────────────────────────┘
```

### The critical problem: update rate

Your model needs **256 ticks at 64Hz** (exactly 4 seconds of data). GSI does NOT guarantee 64 updates per second.

| Situation | GSI updates/second | 256 samples takes... |
|-----------|:---:|:---:|
| Active firefight (lots of state changes) | ~10-20 | 13-26 seconds ❌ |
| With `throttle "0.0"` and `buffer "0.0"` | ~20-30 max (empirical) | 9-13 seconds ❌ |
| Idle / buy phase | 0 (heartbeat only) | ∞ ❌ |
| **What the model needs** | **64** | **4 seconds** ✅ |

> [!WARNING]
> **GSI cannot provide 64Hz tick-level data.** Even at maximum speed (`throttle 0`, `buffer 0`), you'd get maybe 20-30 updates/second during action. This means you cannot fill the model's 256-tick window at the correct temporal resolution. You'd need to either:
> - **Interpolate** the missing ticks (introduces fake data the model wasn't trained on)
> - **Retrain the model** on lower-frequency data (e.g., 10-20Hz instead of 64Hz)
> - **Accept a degraded analysis** with incomplete temporal resolution

### Feature availability (as spectator)

GSI provides this JSON per player when spectating:

```json
{
  "allplayers": {
    "76561198xxxxx": {
      "name": "FriendA",
      "position": "-1234.56, 789.01, 234.56",
      "forward": "0.707, 0.707, 0.0",
      "velocity": "120.0, -45.0, 0.0",
      "state": {
        "health": 100,
        "armor": 100,
        "helmet": true,
        "flashed": 0,
        "money": 4750
      },
      "weapons": {
        "weapon_0": {
          "name": "weapon_ak47",
          "type": "Rifle",
          "ammo_clip": 27,
          "ammo_clip_max": 30,
          "state": "active"
        }
      }
    }
  }
}
```

### Feature mapping: what you CAN vs. CAN'T compute

| Feature | Can compute from GSI? | How / Why not |
|---------|:---:|------|
| `delta_yaw` | ⚠️ Lossy | `atan2(fwd.y, fwd.x)` then diff between updates. **Not per-tick** — temporal resolution too low |
| `delta_pitch` | ⚠️ Lossy | `asin(-fwd.z)` then diff. Same timing issue |
| `target_error_yaw` | ⚠️ Approx | Attacker forward + victim position → angle. Correct math, wrong timing |
| `target_error_pitch` | ⚠️ Approx | Same |
| `shot` | ⚠️ | Detect ammo decrease — misses exact tick |
| `yaw_acceleration` | ⚠️ Very lossy | 2nd derivative of already-lossy yaw |
| `pitch_acceleration` | ⚠️ Very lossy | Same |
| `target_yaw_rate` | ⚠️ | Rate of change of approximate target_error |
| `target_pitch_rate` | ⚠️ | Same |
| `distance` | ✅ | Both positions available |
| `relative_height` | ✅ | Z coordinates available |
| `attacker_speed` | ✅ | Velocity vector available in spectator GSI |
| `victim_speed` | ✅ | Same |
| `closing_speed` | ✅ | Computed from positions + velocities |
| `lateral_speed` | ✅ | Same |
| `attacker_vertical_speed` | ✅ | velocity.z |
| `victim_vertical_speed` | ✅ | Same |
| `scoped` | ❌ | Not in GSI payload |
| `crouch` | ❌ | Not in GSI payload |
| `airborne` | ❌ | Not in GSI payload |
| `flash_remaining` | ⚠️ | Integer `flashed` (0-255), not continuous seconds |
| `health` | ✅ | Available |
| `armor` | ✅ | Available |
| `recoil_index` | ❌ | Not in GSI payload |
| `punch_pitch` | ❌ | Not in GSI payload |
| `punch_yaw` | ❌ | Not in GSI payload |
| `shots_in_burst` | ⚠️ | Infer from ammo changes |
| `ammo` | ✅ | `ammo_clip` available |
| `time_since_shot` | ⚠️ | From ammo change timestamps (imprecise) |
| `victim_shot` | ⚠️ | Detect victim ammo change |
| `victim_footstep` | ❌ | Not in GSI |
| `time_since_victim_noise` | ❌ | Not in GSI |
| `victim_crouch` | ❌ | Not in GSI |
| `victim_health` | ✅ | Available |
| `weapon_auto/sniper/pistol/smg/shotgun` | ✅ | Weapon name → classification |

**Score: ~12 features ✅ reliable, ~12 features ⚠️ approximate/lossy, ~15 features ❌ unavailable**

### GSI cost breakdown

| Item | Cost |
|------|-----:|
| CS2 game (you already own it) | €0 |
| GSI config file | €0 |
| Python HTTP server | €0 |
| **Total** | **€0** |

### GSI verdict

| Aspect | Assessment |
|--------|-----------|
| **Cost** | ✅ Free |
| **Ban risk** | ✅ Zero — Valve-official feature |
| **Setup time** | ✅ ~2-3 hours |
| **Real-time** | ✅ Yes — data pushed live |
| **Feature coverage** | ❌ ~40-50% (12/39 reliable) |
| **Temporal resolution** | ❌ ~10-30 Hz, model needs 64 Hz |
| **Current model works as-is** | ❌ **No** — needs retraining on reduced features + lower Hz |
| **Research/thesis value** | ⚠️ Demonstrates live integration, but accuracy will be lower |

---

## Option 2: Self-Hosted Dedicated Server + CounterStrikeSharp

### What is it
You download and run a CS2 dedicated server (free from Steam). You install Metamod:Source + CounterStrikeSharp on it. You write a C# plugin that captures **all** player data at the server's native 64Hz tick rate and streams it to your Python ML server via WebSocket.

Your friends connect to YOUR server instead of a Valve lobby. They won't notice any difference in gameplay.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│              YOUR MACHINE (or VPS / hosting)              │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │         CS2 DEDICATED SERVER (free)                │  │
│  │                                                    │  │
│  │  Metamod:Source → CounterStrikeSharp               │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │       YOUR C# TELEMETRY PLUGIN               │  │  │
│  │  │                                              │  │  │
│  │  │  OnTick() — 64 times/second:                 │  │  │
│  │  │    For each player:                          │  │  │
│  │  │      EyeAngles.X (pitch), .Y (yaw)           │  │  │
│  │  │      Position (X, Y, Z)                      │  │  │
│  │  │      Velocity (X, Y, Z) — TRUE networked     │  │  │
│  │  │      Health, Armor, DuckAmount               │  │  │
│  │  │      IsScoped, IsAirborne                    │  │  │
│  │  │      AimPunchAngle (pitch, yaw)              │  │  │
│  │  │      RecoilIndex                             │  │  │
│  │  │      ActiveWeapon, Ammo                      │  │  │
│  │  │                                              │  │  │
│  │  │  OnPlayerHurt() — encounter anchor           │  │  │
│  │  │  OnWeaponFire() — shot events                │  │  │
│  │  │  OnPlayerBlind() — flash events              │  │  │
│  │  │                                              │  │  │
│  │  │  → JSON via WebSocket to Python              │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                               │
│                     WebSocket                            │
│                          ▼                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │          PYTHON INFERENCE SERVER                   │  │
│  │                                                    │  │
│  │  1. Receive 64Hz tick data per player              │  │
│  │  2. Ring buffer: [10 players × 256 ticks × 20+    │  │
│  │     properties]                                    │  │
│  │  3. On player_hurt: extract [256, 39] window      │  │
│  │  4. cs2_data.py feature computation (REUSED)       │  │
│  │  5. TCN + LightGBM inference (REUSED)              │  │
│  │  6. Terminal: score + color + details              │  │
│  │                                                    │  │
│  │  Latency: < 200ms from kill to score display      │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Feature completeness: 39/39

| Feature | Available | Source |
|---------|:---------:|--------|
| `delta_yaw` | ✅ | `EyeAngles.Y` diff per tick |
| `delta_pitch` | ✅ | `EyeAngles.X` diff per tick |
| `target_error_yaw` | ✅ | Attacker eye angles + victim position |
| `target_error_pitch` | ✅ | Same |
| `shot` | ✅ | `weapon_fire` event |
| `yaw_acceleration` | ✅ | 2nd derivative of exact yaw |
| `pitch_acceleration` | ✅ | Same |
| `target_yaw_rate` | ✅ | Rate of target_error change |
| `target_pitch_rate` | ✅ | Same |
| `distance` | ✅ | Euclidean from positions |
| `relative_height` | ✅ | Z + eye_height difference |
| `attacker_speed` | ✅ | **True velocity** (server-side, not derived) |
| `victim_speed` | ✅ | Same |
| `closing_speed` | ✅ | Radial component of relative velocity |
| `lateral_speed` | ✅ | Tangential component |
| `attacker_vertical_speed` | ✅ | velocity.Z |
| `victim_vertical_speed` | ✅ | Same |
| `scoped` | ✅ | `m_bIsScoped` entity property |
| `crouch` | ✅ | `m_flDuckAmount` entity property |
| `airborne` | ✅ | `IsAirborne` or ground entity check |
| `flash_remaining` | ✅ | `player_blind` event + continuous decay |
| `health` | ✅ | Entity property |
| `armor` | ✅ | Entity property |
| `recoil_index` | ✅ | `m_flRecoilIndex` entity property |
| `punch_pitch` | ✅ | `m_aimPunchAngle[0]` |
| `punch_yaw` | ✅ | `m_aimPunchAngle[1]` |
| `shots_in_burst` | ✅ | Counter from weapon_fire events |
| `ammo` | ✅ | `m_iClip1` entity property |
| `time_since_shot` | ✅ | Tick count since last weapon_fire |
| `victim_shot` | ✅ | Same for victim |
| `victim_footstep` | ✅ | `player_footstep` event |
| `time_since_victim_noise` | ✅ | Tick count since footstep/shot |
| `victim_crouch` | ✅ | Victim's duck_amount |
| `victim_health` | ✅ | Victim entity property |
| `weapon_*` one-hot | ✅ | Weapon name classification |

**Score: 39/39 features at full 64Hz fidelity.** This matches the training data distribution exactly.

### Three hosting tiers

#### Tier A: Self-host on your PC (Free)

Run the CS2 dedicated server on the same PC you're playing on.

| Requirement | Details |
|-------------|---------|
| CPU | 6+ cores recommended (server uses ~2, game uses ~4) |
| RAM | 16GB+ (server ~3-4GB, game ~6-8GB, OS + Python ~2GB) |
| Storage | ~65GB for server files (separate from your game install) |
| Network | Port forward 27015 (UDP/TCP) on your router |
| Your friends connect via | `connect YOUR_PUBLIC_IP:27015` in console |

| Item | Cost |
|------|-----:|
| SteamCMD + CS2 Dedicated Server | €0 (free download) |
| Metamod:Source | €0 (open source) |
| CounterStrikeSharp | €0 (open source) |
| GSLT Token (Steam account) | €0 (free) |
| Electricity for your PC | ~€0.10/session |
| **Total** | **€0** |

**Pros:** Free. Lowest latency for you.
**Cons:** Your friends get higher ping (connecting to your home internet). Your game performance may drop if your PC isn't powerful enough for both. Requires port forwarding. Your IP is exposed.

#### Tier B: VPS (Cheapest remote hosting)

Rent a small Linux VPS. Install CS2 server + Metamod + CSS yourself.

| Provider | Plan | Specs | Monthly Cost |
|----------|------|-------|-------------:|
| **Hetzner** CX23 | 2 vCPU, 4GB RAM, 40GB SSD | EU datacenter | **~€6/month** |
| **Hetzner** CPX22 | 2 vCPU, 4GB RAM, 40GB SSD | Better CPU | **~€8/month** |
| **Vultr** | 2 vCPU, 4GB RAM, 80GB SSD | Global DCs | **~$12/month** |
| **DigitalOcean** | 2 vCPU, 4GB RAM, 80GB SSD | Global DCs | **~$12/month** |

| Item | Cost |
|------|-----:|
| VPS (Hetzner CX23) | ~€6/month |
| CS2 Dedicated Server | €0 |
| Metamod + CSS | €0 |
| **Total** | **~€6/month** |

**Pros:** Good ping for everyone (EU datacenter). Your PC is free for gaming. Always-on if needed.
**Cons:** You manage Linux yourself. Need to learn SteamCMD + server admin. Storage might need 80GB (some plans need upgrade).

> [!WARNING]
> **Storage concern:** The CS2 dedicated server is ~60-65GB. The cheapest Hetzner plan has 40GB SSD. You may need the 80GB plan (**~€8-10/month**) or attach additional block storage.

#### Tier C: Managed Game Hosting (Easiest)

Use a service like DatHost that offers one-click Metamod + CounterStrikeSharp support.

| Provider | Pricing Model | Estimated Cost (10 slots) | CSS Support |
|----------|---------------|:---:|:---:|
| **DatHost** | ~€0.99/slot/month | **~€10/month** | ✅ One-click |
| **GravelHost** | Flat plans | **~€5-7/month** | ⚠️ Manual |
| **Host Havoc** | Flat plans | **~€8-10/month** | ⚠️ Manual |

| Item | Cost |
|------|-----:|
| DatHost 10-slot server | ~€10/month |
| Metamod + CSS | €0 (one-click install on DatHost) |
| **Total** | **~€10/month** |

**Pros:** One-click setup. No Linux knowledge needed. Web control panel. Auto-updates. Good performance. DatHost specifically supports CSS out of the box.
**Cons:** Most expensive option. Less control than VPS.

### Self-hosted server cost summary

| Hosting Tier | Monthly Cost | Setup Difficulty | Performance |
|-------------|-------------:|:---:|:---:|
| **A: Your PC** | **€0** | Medium | ⚠️ Depends on your hardware |
| **B: VPS (Hetzner)** | **€6-10** | Hard (Linux admin) | ✅ Good |
| **C: Managed (DatHost)** | **€10-15** | Easy (web panel) | ✅ Great |

---

## Head-to-Head Comparison

| Criterion | Option 1: GSI Spectator | Option 2: Self-Hosted Server |
|-----------|:---:|:---:|
| **Monthly cost** | **€0** | €0-15 depending on tier |
| **Setup time** | ~3 hours | 4-8 hours (+ plugin dev 1-2 days) |
| **Ban risk** | ✅ Zero | ✅ Zero |
| **You spectate** | ✅ On Valve server | ✅ On your server (or play) |
| **Friends' experience** | ✅ Normal Valve match | ✅ Normal (just different IP) |
| **Feature coverage** | ❌ 12/39 reliable | ✅ **39/39** |
| **Temporal resolution** | ❌ ~10-30 Hz | ✅ **64 Hz** |
| **Current model works** | ❌ Must retrain | ✅ **Works as-is** |
| **Real-time latency** | ✅ ~100-500ms | ✅ **< 200ms** |
| **Data quality** | ⚠️ Approximate | ✅ **Exact match to training data** |
| **GOTV demo also available** | ❌ Can't record on Valve server | ✅ Can `tv_record` on your server |
| **Thesis value** | ⚠️ Proof of concept | ✅ **Full live evaluation chapter** |

---

## Full Cost Summary

### Option 1: GSI Spectator

| Cost Category | One-Time | Monthly | Notes |
|---------------|:--------:|:-------:|-------|
| Hardware | €0 | €0 | Your existing PC |
| Software | €0 | €0 | CS2 (owned) + Python (free) |
| Hosting | €0 | €0 | Valve hosts the server |
| Development time | ~3 hours | — | GSI listener + feature computation |
| Model retraining | ~4-8 hours | — | Need reduced-feature model |
| **Total** | **~8-12 hours labor** | **€0/month** | |

### Option 2: Self-Hosted Server

| Cost Category | One-Time | Monthly | Notes |
|---------------|:--------:|:-------:|-------|
| **If self-hosting on PC:** | | | |
| Hardware | €0 | €0 | Your existing PC (needs 16GB+ RAM, 6+ cores) |
| Software | €0 | €0 | All free/open-source |
| CS2 Dedicated Server | €0 | €0 | Free via SteamCMD |
| Metamod + CSS | €0 | €0 | Open source |
| Development (C# plugin) | ~8-16 hours | — | Telemetry capture + WebSocket streaming |
| Development (Python listener) | ~4-6 hours | — | WebSocket receiver + feature computation |
| **Subtotal (self-host)** | **~12-22 hours labor** | **€0/month** | |
| | | | |
| **If using VPS (Hetzner):** | | | |
| Same as above, plus: | | | |
| VPS rental | — | **€6-10/month** | Hetzner CX23 or CPX22 |
| **Subtotal (VPS)** | **~12-22 hours labor** | **€6-10/month** | |
| | | | |
| **If using managed hosting (DatHost):** | | | |
| Same as above, minus Linux setup: | | | |
| Managed hosting | — | **€10-15/month** | One-click CSS support |
| **Subtotal (managed)** | **~10-18 hours labor** | **€10-15/month** | |

---

## My Recommendation

For your thesis scenario (spectate friends, analyze kills, demonstrate live ML detection):

> [!TIP]
> **Start with Option 2, Tier A (self-host on your PC, €0/month).**
> 
> - It's free
> - You get 100% feature coverage at 64Hz
> - Your existing TCN+LightGBM model works without retraining
> - Your friends just `connect` to your IP instead of using a lobby code
> - You can also `tv_record` GOTV demos for offline analysis later
> - The thesis chapter writes itself: "Live Server-Side Detection Evaluation"
> 
> If your PC can't handle running both (server + game client), or your friends' ping is bad, upgrade to a VPS (€6-10/month) or DatHost (€10-15/month).

GSI (Option 1) is a cool proof-of-concept but it fundamentally can't feed the model correctly due to the temporal resolution gap. You'd spend more time retraining the model than setting up a server.

---

## What needs to be built (Option 2)

| Component | Language | LOC Estimate | Reuses existing code? |
|-----------|----------|:---:|:---:|
| C# telemetry plugin | C# | ~400-600 | No (new) |
| Python WebSocket listener | Python | ~200-300 | No (new) |
| Feature computation | Python | ~100-150 | ✅ Reuses `cs2_data.py` helpers |
| Model inference | Python | ~50-80 | ✅ Reuses `private_scoring.py` |
| Terminal output UI | Python | ~80-120 | ✅ Reuses `format_simple_report()` |
| **Total new code** | | **~800-1250** | ~40% reused from existing pipeline |

Would you like me to start building any of these components?

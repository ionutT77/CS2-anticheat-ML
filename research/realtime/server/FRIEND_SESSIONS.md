# Friend sessions: evaluation-group protocol (v1, draft)

Pre-registered protocol for collecting **legitimate** CS2 data from consenting friends on the
private server. Fixed before any session is recorded, so results cannot steer the design.
Status: draft, revised after a design review (journal Entry 27). Consent wording still to be
approved by the supervisor / ethics process.

## Purpose and role of the data

- **Role: evaluation group only.** These sessions measure the false-positive rate of the
  frozen model (`experiments/cs2cd_v1`) on known-legitimate players, by weapon style. They
  are **not** training data, and no threshold is chosen from them.
- **Main question:** for the same people, how much more often are scout-heavy and
  AWP-heavy play flagged than normal play, at each existing threshold?
- **Why:** on the CS2CD test split all 4 strict-threshold false positives were scout-heavy
  legitimate players (4/25). Legitimate pro AWPers are flagged at the lenient thresholds
  (13/100 at `accuracy`, 21/100 at `f1`; 0/100 at `fpr_1pct`). See journal Entry 27.

## Design: every person plays every mode

| Mode | Server setup | Role |
|---|---|---|
| **Normal** | standard competitive | baseline and **domain-shift control** (friends, private server) |
| **Scout-only** | everyone has an SSG 08, buying disabled | scout-heavy play |
| **AWP-only** | everyone has an AWP, buying disabled | AWP-heavy play |

- **Within-person design:** each participant plays all three modes. Results are analysed as
  differences **within each person** (scout-only minus normal, AWP-only minus normal), so a
  difference between modes cannot come from different people.
- **Counterbalanced order:** the order of modes is rotated between sessions (e.g. N-S-A,
  S-A-N, A-N-S), assigned before the first session.
- Setup to check on the server before the first session (not yet verified in CS2):
  `mp_ct_default_primary weapon_ssg08` or `weapon_awp`, the same for
  `mp_t_default_primary`, and `mp_buytime 0`. **Never** use `mp_warmup_end` or
  `mp_restartgame`, which cut the GOTV recording.
- **Humans only, standard timings:** start the server with
  `start_server.ps1 -LanIp <ip> -RconPassword <pw> -BotQuota 0 -Competitive`. `-Competitive`
  removes the bot-test shortcuts (2 s freeze time, 3 min rounds) so normal mode is a normal
  match. Check after each session that no bot appears in the capture. The script already
  starts without a warmup period, so no warmup command is needed.

## Sample size and stopping rule (pre-registered)

- **Unit:** one player-match = one person in one match (one map, from the first live round
  to the end of the map). It counts only if it has **≥ 5 eligible encounters**.
- **Target:** at least **12 distinct people**, more is better; each plays **up to 6
  player-matches per mode**. If a person plays more, only their **first 6 per mode, in time
  order**, are used. This rule is fixed now.
- **Stopping rule:** collection closes on a date fixed before the first session (or when
  the per-mode target is reached, whichever is first). Sessions are never added or dropped
  because of scores.
- **The effective sample size is the number of people, not the number of matches.**
  Matches from one person are not independent. Exact intervals that treat 72 player-matches
  as independent (0 flagged gives 0–5.0%) are shown only as a secondary figure.

## Blind collection

- **Nobody looks at scores during the collection period.** The listener is not used during
  these sessions, or only to write its log without anyone reading it.
- All results come from **offline scoring after collection closes** (see analysis plan).
  Live flags are not used: they can differ from offline scoring and are provisional below
  10 encounters.

## Checklist for each session

**Before**

- [ ] Every player has given **written consent**: what is recorded (positions, view angles,
      shots, hits, health, weapon, SteamID64), that it is for a thesis, that data stays
      private and is published only in aggregate, and that they can withdraw.
- [ ] Every player's SteamID64 is in `consent.txt`; **restart the server after editing it**.
- [ ] Mode order for this session taken from the pre-assigned rotation.
- [ ] Server started with `-BotQuota 0 -Competitive`; mode cvars set.
- [ ] Note the plugin version, `git rev-parse HEAD`, and the CS2 build (`status` → version).

**During**

- [ ] "Play normally"; no instructions about aiming or behaviour.
- [ ] No warmup or restart commands; no score viewing.

**After**

- [ ] `tv_stoprecord`, then `quit`.
- [ ] Keep capture, GOTV demo and server log together; fill in the metadata row.
- [ ] Check that each match used the mode's weapon (weapon counts in the capture) and that
      no bot appears.
- [ ] Count player-matches per person per mode against the cap.

## Per-session metadata (one row per session)

| Field | Example |
|---|---|
| session_id | S2026-11-02-a |
| date | 2026-11-02 |
| mode order | N-S-A |
| map per match | de_mirage, de_inferno, de_dust2 |
| players | P01, P02, P05, P07 (aliases; SteamIDs only in the private consent record) |
| plugin version | 0.2.0-schema1 |
| code revision | git commit hash |
| CS2 build | from `status` |
| server launch command / config hash | parameters used |
| model + threshold files | SHA-256 of `selection.json` and `calibration.json` |
| tickrate | 64 (capture header) |
| files | capture, GOTV demo, server log |
| encounters per player-match | filled in after offline extraction |
| notes | disconnects, server problems |

Keep the alias-to-SteamID mapping and signed consent forms **outside the repository**.

## Analysis plan (fixed now)

1. After collection closes, build windows with `realtime/capture_windows.py` (no bot
   victims) and score each player-match offline with the frozen model, all thresholds
   unchanged.
2. **Primary result:** per mode, the number of **people flagged at least once / N people**,
   and the per-person flag rate averaged with **equal weight per person**, with a 95%
   bootstrap interval that resamples **people**.
3. **Paired comparison:** scout-only vs normal and AWP-only vs normal within each person.
4. **Secondary:** player-match false-positive rate with exact Clopper-Pearson intervals.
5. All thresholds: `accuracy`, `f1`, `default_0_5`, `fpr_1pct`, `fpr_0_1pct`.
6. No threshold or model change is made from this data.

## Reuse

- Every use of this group to evaluate a model is logged in the journal.
- Before the first analysis, **one third of the people are set aside** as a final hold-out,
  used only once, for the final model of the thesis. The rest may be reused for
  development, which then makes them a development set rather than an independent test.

## Limits

- Friends are not a random sample of players: skill, style and number are limited.
- **Normal mode is the domain-shift control.** Friends on a private server differ from
  CS2CD matchmaking (skill, opponents, setting); a raised rate in normal mode reflects that
  shift, not the weapon.
- Scout-only and AWP-only matches are unusual games (everyone uses the same weapon, so
  victim behaviour differs). They measure style-specific false positives, not live
  performance.
- No cheating players are present, so recall and precision cannot be measured here.

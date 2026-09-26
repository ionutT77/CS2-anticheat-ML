"""Score a private-server CS2 match: .dem → extraction → inference → report.

Shadow/review mode only.  This script never bans, kicks or punishes players.
It is designed for authorized private-server research and thesis evaluation.
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = RESEARCH_ROOT / 'experiments' / 'cs2cd_v1'
sys.path.insert(0, str(RESEARCH_ROOT))
from anticheat.cs2_inference import predict
from anticheat.dem_extractor import extract_dem
from anticheat.private_scoring import (
    ENCOUNTER_FIELDS, EVENT_SCORE_DESCRIPTION, add_feature_audit, predict_encounters,
)

_BANNER = """\
================================================================================
  CS2CD PRIVATE-SERVER MATCH REPORT — SHADOW / REVIEW MODE
================================================================================

  ⚠  This system is for authorized research and thesis evaluation ONLY.
  ⚠  It does NOT ban, kick, or punish any player.
  ⚠  Frozen CS2CD benchmark ROC-AUC ≈ 0.973 does NOT predict live-server
     accuracy.  Scores must be interpreted with caution.
  ⚠  All participants must have given explicit consent.
================================================================================
"""


def format_report(audit, scores, output_dir, encounter_scores=()):
    """Build a human-readable text report."""
    lines = [_BANNER]
    header = audit.get('header', {})
    lines.append(f"Demo:       {audit.get('dem_path', 'unknown')}")
    lines.append(f"Map:        {header.get('map_name', header.get('map', 'unknown'))}")
    lines.append(f"Players:    {len(audit.get('players', []))}")
    lines.append(f"Encounters: {audit.get('encounter_count', 0)}")
    lines.append(f"Anchors:    {audit.get('anchor_count', 0)}")
    lines.append('')

    # Rejection summary
    rejected = audit.get('rejected', {})
    if rejected:
        lines.append('Rejection Summary:')
        for reason, count in sorted(rejected.items()):
            lines.append(f'  {reason}: {count}')
        lines.append('')

    # Per-player results table
    if scores:
        lines.append('Per-Player Results:')
        lines.append('─' * 80)
        hdr = f"  {'Player':<12} {'Enc':>4}  {'Raw Score':>10}  {'Calibrated':>10}  Flags"
        lines.append(hdr)
        lines.append('─' * 80)
        for row in scores:
            flags = [k.removeprefix('flag_') for k, v in row.items()
                     if k.startswith('flag_') and v]
            flag_str = ', '.join(flags) if flags else 'none'
            lines.append(
                f"  {row['player']:<12} {row['encounters']:>4}  "
                f"{row['raw_score']:>10.6f}  {row['calibrated_score']:>10.6f}  {flag_str}"
            )
        lines.append('')
    else:
        lines.append('No players could be scored (no eligible encounters).\n')

    lines.append('Per-Encounter Neural Evidence:')
    lines.append(EVENT_SCORE_DESCRIPTION)
    for row in encounter_scores:
        lines.append(
            f"  #{row['encounter_index']} {row['player']} tick={row['tick']} "
            f"victim={row['victim']} weapon={row['weapon']} component={row['component']} "
            f"event_logit={row['event_logit']:.6f} event_score={row['event_score']:.6f}"
        )
    if not encounter_scores:
        lines.append('  No neural encounter scores available.')
    lines.append('')

    lines.append('Known Approximations:')
    for approximation in audit.get('known_approximations', []):
        lines.append(f'  {approximation}')
    lines.append('')
    lines.append('Feature Statistics:')
    lines.append(audit.get('feature_stats_scope', ''))
    for feature, stats in audit.get('feature_stats', {}).items():
        lines.append(f'  {feature}: ' + ', '.join(f'{key}={value}' for key, value in stats.items()))
    lines.append('')

    # Players with no encounters
    no_enc = [p for p in audit.get('players', []) if p['encounters'] == 0]
    if no_enc:
        lines.append('Players with no eligible encounters (cannot be scored):')
        for p in no_enc:
            lines.append(f"  {p['local_id']} — {p['total_ticks']} ticks recorded")
        lines.append('')

    # Approximated features
    approx = audit.get('approximated_features', [])
    if approx:
        lines.append('Approximated Features (not silently zeroed — see audit JSON):')
        for feat in approx:
            lines.append(f'  {feat}')
        lines.append('')
    else:
        lines.append('Approximated Features: none')
        lines.append('')

    # Warnings
    warns = audit.get('warnings', [])
    if warns:
        lines.append('Warnings:')
        for w in warns:
            lines.append(f'  {w}')
        lines.append('')

    # Output files
    lines.append(f'Output Files ({output_dir}):')
    lines.append('  encounters.npz          — NPZ input for predict_cs2cd.py')
    lines.append('  scores.csv              — per-player ensemble/calibrated scores')
    lines.append('  encounter_scores.csv    — uncalibrated neural event-head evidence')
    lines.append('  extraction_audit.json   — extraction log and raw feature statistics')
    lines.append('  report.txt              — this report')
    lines.append('  simple_report.txt       — simplified non-technical player verdicts')
    lines.append('')

    lines.append('─' * 80)
    lines.append('NOTE: The frozen CS2CD benchmark AUC ≈ 0.973 was measured on a')
    lines.append('specific dataset with specific labels. It does not imply 97.3%')
    lines.append('accuracy, nor does it predict performance on this private server.')
    lines.append('Interpret all scores as exploratory research results.')
    lines.append('─' * 80)
    return '\n'.join(lines)


def format_simple_report(audit, scores, encounter_scores, sid_to_name, extreme_thresh, ban_ratio, sus_ratio):
    lines = []
    
    # Organize encounters by player
    player_encs = {}
    for enc in encounter_scores:
        p = enc['player']
        if p not in player_encs:
            player_encs[p] = []
        player_encs[p].append(enc)
    
    # Map local_id to name
    lid_to_name = {}
    for p in audit.get('players', []):
        sid = p['steamid']
        name = sid_to_name.get(str(sid), "Unknown")
        lid_to_name[p['local_id']] = name

    for row in scores:
        p = row['player']
        name = lid_to_name.get(p, "Unknown")
        encs = player_encs.get(p, [])
        total_shots = len(encs)
        
        encs = sorted(encs, key=lambda x: x['tick'])
        
        extreme_shots = [e for e in encs if e['event_score'] >= extreme_thresh]
        num_extreme = len(extreme_shots)
        
        flagged_shots = [e for e in encs if e['event_score'] >= 0.5]
        num_flagged = len(flagged_shots)
        
        ratio = num_extreme / total_shots if total_shots > 0 else 0.0
        flagged_ratio = num_flagged / total_shots if total_shots > 0 else 0.0
        conf_pct = int(row['calibrated_score'] * 100)
        
        if ratio >= ban_ratio and num_extreme >= 3:
            status = "He is for sure cheating - insta ban"
        elif flagged_ratio >= sus_ratio or row.get('flag_accuracy') or row.get('flag_f1'):
            status = "suspicious - manual review needed"
        else:
            status = "clean"
            
        lines.append(f'{p}: "{status}, confidence score {conf_pct} %" - {name}')
        
        if total_shots > 0:
            lines.append("All shots analysed:")
            for e in encs:
                tag = "🔴 EXTREME" if e['event_score'] >= extreme_thresh else ("🟡 Suspicious" if e['event_score'] >= 0.5 else "⚪ Normal")
                v_name = lid_to_name.get(e['victim'], e['victim'])
                lines.append(f"  - Tick {e['tick']:>7} vs {v_name:<15} | Weapon: {e['weapon']:<8} | Score: {e['event_score']:.4f} {tag}")
            lines.append("")
        else:
            lines.append("")
            
    return '\n'.join(lines)


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    ap = argparse.ArgumentParser(
        description='Score a private-server CS2 match (shadow/review mode).',
        epilog='This tool does NOT ban, kick or punish any player.',
    )
    ap.add_argument('demo', type=Path, help='Path to the CS2 .dem file.')
    ap.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR,
                    help='Model artifacts (default: research/experiments/cs2cd_v1 relative to this script).')
    ap.add_argument('--output', type=Path, required=True,
                    help='Output directory for report, NPZ and scores.')
    ap.add_argument('--player', help='Score one match-local player ID only.')
    ap.add_argument('--consent-ids', nargs='*',
                    help='Steam64 IDs of consenting participants (optional filter).')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--prop-map', type=Path,
                    help='JSON file mapping internal property names to demoparser2 names.')
    ap.add_argument('--ban-ratio', type=float, default=0.7,
                    help='Ratio of extreme shots to total to trigger insta-ban (default: 0.7)')
    ap.add_argument('--sus-ratio', type=float, default=0.35,
                    help='Ratio of flagged (>=0.5) shots to trigger manual review (default: 0.35)')
    ap.add_argument('--extreme-threshold', type=float, default=0.85,
                    help='Threshold for individual shot to be extreme (default: 0.85)')
    args = ap.parse_args()
    if args.consent_ids is not None and not args.consent_ids:
        ap.error('--consent-ids received no Steam64 IDs; pass IDs or omit the flag.')
    consent = set(args.consent_ids) if args.consent_ids is not None else None

    # UTF-8 files and redirected console output are portable on Windows as well.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print(_BANNER)

    prop_map = None
    if args.prop_map is not None:
        prop_map = json.loads(args.prop_map.read_text(encoding='utf-8'))

    # Step 1: Extract encounters from demo
    args.output.mkdir(parents=True, exist_ok=True)
    npz_path = args.output / 'encounters.npz'

    torch.set_num_threads(6)
    audit = extract_dem(
        args.demo,
        output_path=npz_path,
        consent_players=consent,
        prop_map=prop_map,
    )

    with np.load(npz_path, allow_pickle=False) as encounters:
        x, players, ticks = encounters['x'], encounters['player'].astype(str), encounters['tick']
    add_feature_audit(audit, x)
    events = audit.get('events', [])
    if len(events) != len(x):
        raise ValueError("audit['events'] must align with extracted NPZ encounters.")
    selected_indices = np.arange(len(x))
    if args.player is not None:
        known_players = {player['local_id'] for player in audit.get('players', [])}
        if args.player not in known_players:
            ap.error(f'Unknown player {args.player!r}; available IDs: {", ".join(sorted(known_players)) or "none"}.')
        selected_indices = np.flatnonzero(players == args.player)
        if not len(selected_indices):
            ap.error(f'Player {args.player!r} has no eligible encounters.')

    # Step 2: Keep player-level inference unchanged; event evidence is separate.
    scores, encounter_scores = [], []
    if len(selected_indices):
        selected_x = x[selected_indices]
        selected_players, selected_ticks = players[selected_indices], ticks[selected_indices]
        selected_events = [events[index] for index in selected_indices]
        encounter_scores = predict_encounters(
            selected_x, selected_players, selected_ticks, selected_events, args.model_dir, args.device)
        for row in encounter_scores:
            row['encounter_index'] = int(selected_indices[row['encounter_index']])
        scores = predict(selected_x, selected_players, selected_ticks, args.model_dir, args.device)
    audit['scoring_player_filter'] = args.player
    audit['scored_encounter_count'] = len(selected_indices)

    # Step 3: Write outputs
    # CSV scores
    scores_path = args.output / 'scores.csv'
    if scores:
        with scores_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
            writer.writeheader()
            writer.writerows(scores)
        print(f'Wrote {len(scores)} player scores to {scores_path}')
    else:
        scores_path.write_text('# No eligible encounters\n', encoding='utf-8')

    with (args.output / 'encounter_scores.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=ENCOUNTER_FIELDS)
        writer.writeheader()
        writer.writerows(encounter_scores)

    # Serialize the enriched in-memory audit, not the extractor's earlier snapshot.
    audit_dst = args.output / 'extraction_audit.json'
    audit_dst.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

    sid_to_name = {}
    try:
        from demoparser2 import DemoParser
        pi = DemoParser(str(args.demo)).parse_player_info()
        sid_to_name = {str(r['steamid']): r['name'] for _, r in pi.iterrows()}
    except Exception as e:
        logging.warning(f"Could not parse player names from demo: {e}")

    report = format_report(audit, scores, args.output, encounter_scores)
    report_path = args.output / 'report.txt'
    report_path.write_text(report, encoding='utf-8')
    
    simple_report = format_simple_report(audit, scores, encounter_scores, sid_to_name, args.extreme_threshold, args.ban_ratio, args.sus_ratio)
    simple_report_path = args.output / 'simple_report.txt'
    simple_report_path.write_text(simple_report, encoding='utf-8')

    print(report)
    print("\n" + "="*80)
    print("  SIMPLIFIED REPORT (simple_report.txt)")
    print("="*80 + "\n")
    print(simple_report)


if __name__ == '__main__':
    main()

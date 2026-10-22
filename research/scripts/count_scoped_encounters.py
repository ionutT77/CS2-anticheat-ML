"""Count scoped-weapon encounters by label for CS2CD and the staged OpenCS2 data.

Read-only: reproduces the tables in research/opencs2/SCOUT_AWP_BALANCE.md.
Run from `research/`:  python scripts/count_scoped_encounters.py
An encounter is the first hit of an attacker on a victim in a burst (not a kill). Labels are
player-match labels: "cheating" means the player was labelled a cheater.
"""
import pandas as pd

GROUPS = {'awp': 'AWP', 'ssg08': 'Scout (SSG 08)', 'scar20': 'Auto-sniper (SCAR-20/G3SG1)',
          'g3sg1': 'Auto-sniper (SCAR-20/G3SG1)', 'aug': 'Scoped rifle (AUG/SG 553)',
          'sg556': 'Scoped rifle (AUG/SG 553)'}
ORDER = ['AWP', 'Scout (SSG 08)', 'Auto-sniper (SCAR-20/G3SG1)', 'Scoped rifle (AUG/SG 553)']
COLUMNS = ['split', 'label', 'weapon']


def load(path):
    d = pd.read_parquet(path, columns=COLUMNS)
    d['group'] = d['weapon'].astype(str).str.removeprefix('weapon_').str.lower().map(GROUPS)
    return d[d.group.notna()]


def table(d):
    t = d.groupby(['group', 'label']).size().unstack(fill_value=0).reindex(ORDER).fillna(0).astype(int)
    for label in (0, 1):
        if label not in t:
            t[label] = 0
    return t.rename(columns={0: 'not_cheating', 1: 'cheating'})[['cheating', 'not_cheating']]


def main():
    cs2cd = load('experiments/cs2cd_v1/events.parquet')
    addition = load('data/opencs2/awp_v1/events.parquet')
    reserve = load('data/opencs2/awp_v1/reserve_events.parquet')
    train = table(cs2cd[cs2cd.split == 'train'])
    print('CS2CD, training split\n', train.to_string())
    print('\nCS2CD, all splits\n', table(cs2cd).to_string())
    print('\nOpenCS2 training addition (legitimate)\n', table(addition).to_string())
    print('\nOpenCS2 reserve (legitimate)\n', table(reserve).to_string())
    gap = (train['cheating'] - train['not_cheating']).clip(lower=0)
    print('\nlegitimate encounters needed for 1:1 in training (before any addition)\n', gap.to_string())


if __name__ == '__main__':
    main()

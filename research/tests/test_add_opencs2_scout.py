from scripts.add_opencs2_scout import accepted_player, ranked_demos, scout_kills, yield_exhausted


def test_yield_rule_uses_only_the_last_usable_demos():
    demos = [{'scout_encounters': n} for n in (9, 8, 2, 2, 1, 2, 2)] + [{'error': 'x'}]
    assert yield_exhausted(demos, window=5, threshold=2)       # 2, 2, 1, 2, 2 -> 1.8
    assert not yield_exhausted(demos[:5], window=5, threshold=2)   # 9, 8, 2, 2, 1 -> 4.4
    assert not yield_exhausted(demos[:3], window=5, threshold=2)   # fewer demos than the window


def row(name, scout, other=0):
    return {'file_name': name, 'kills': [{'weapon': 'ssg08'}] * scout + [{'weapon': 'ak47'}] * other}


def test_scout_kills_ignores_other_weapons():
    assert scout_kills(row('a', 3, other=10)) == 3
    assert scout_kills({'file_name': 'b', 'kills': None}) == 0


def test_ranking_is_by_scout_kills_stable_and_skips_excluded_and_zero():
    rows = [row('a', 1), row('b', 4), row('c', 0, other=9), row('d', 4), row('e', 2)]
    assert [r['file_name'] for r in ranked_demos(rows, exclude={'e'})] == ['b', 'd', 'a']


def test_acceptance_needs_min_scout_encounters():
    events = [{'weapon': 'ssg08'}] * 2 + [{'weapon': 'ak47'}] * 30
    assert not accepted_player(events, 3)
    assert accepted_player(events + [{'weapon': 'ssg08'}], 3)

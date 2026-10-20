from concurrent.futures import ThreadPoolExecutor

from scripts.add_opencs2_awp import candidates, ordered_demos, settle


def test_failed_download_is_reported_not_raised():
    def broken():
        raise IOError('size mismatch for demos/x.dem')
    with ThreadPoolExecutor(max_workers=1) as ex:
        result, error = settle(ex.submit(broken))
    assert result is None and error.startswith('OSError') and 'size mismatch' in error


def test_successful_download_passes_through():
    with ThreadPoolExecutor(max_workers=1) as ex:
        result, error = settle(ex.submit(lambda: 'path.dem'))
    assert result == 'path.dem' and error is None


def kills(spec):
    """spec: {steamid: (total kills, sniper kills)}"""
    out = []
    for sid, (total, sniper) in spec.items():
        out += [{'attacker_steamid': sid, 'weapon': 'awp'}] * sniper
        out += [{'attacker_steamid': sid, 'weapon': 'ak47'}] * (total - sniper)
    return out


def test_candidate_rule_needs_8_kills_and_half_sniper():
    row = {'kills': kills({1: (8, 4), 2: (7, 7), 3: (10, 4), 4: (12, 9)})}
    assert candidates(row) == ['1', '4']


def test_ordering_is_deterministic_and_puts_two_candidate_demos_first():
    rows = [{'file_name': f'd{i}', 'kills': kills({10 * i: (8, 8), 10 * i + 1: (8, 8 if i % 2 else 0)})}
            for i in range(10)]
    first = [r['file_name'] for r, _ in ordered_demos(rows, seed=1)]
    assert first == [r['file_name'] for r, _ in ordered_demos(rows, seed=1)]
    counts = [len(c) for _, c in ordered_demos(rows, seed=1)]
    assert counts == sorted(counts, reverse=True)

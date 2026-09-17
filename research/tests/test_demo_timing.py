"""Synthetic CS2 container tests: no real demos, downloads or decompression."""
import builtins
import struct

import pytest

from anticheat import demo_timing
from anticheat.demo_timing import read_demo_timing

_PREFIX = b'PBDEMS2\0' + bytes(8)


def _varint(value):
    encoded = bytearray()
    while value >= 128:
        encoded.append((value & 127) | 128)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _frame(command, payload=b'', tick=0):
    return _varint(command) + _varint(tick) + _varint(len(payload)) + payload


def _timing(seconds=10., ticks=640):
    return b'\x0d' + struct.pack('<f', seconds) + b'\x10' + _varint(ticks & ((1 << 64) - 1))


def _demo(tmp_path, frames, prefix=_PREFIX):
    path = tmp_path / 'synthetic.dem'
    path.write_bytes(prefix + frames)
    return path


@pytest.mark.parametrize('rate', [64, 128])
@pytest.mark.parametrize('after_stop', [False, True])
def test_timing_values(tmp_path, rate, after_stop):
    # Unknown protobuf fields exercise skipping without decoding game_info.
    extra = b'\x18\x01\x22\x03abc\x29' + bytes(8) + b'\x35' + bytes(4)
    frames = _frame(1, b'ignored', tick=0xffffffff) + _frame(7 | 64, b'not decoded')
    if after_stop:
        frames += _frame(0) + _frame(15, b'spawn groups')
    frames += _frame(2, extra + _timing(10.5, int(rate * 10.5)))
    assert read_demo_timing(_demo(tmp_path, frames)) == {
        'playback_time': 10.5,
        'playback_ticks': int(rate * 10.5),
        'timing_source': 'CDemoFileInfo',
    }


@pytest.mark.parametrize('seconds,ticks', [(0., 0), (-1., -64)])
def test_values_are_not_rate_validated(tmp_path, seconds, ticks):
    result = read_demo_timing(_demo(tmp_path, _frame(2, _timing(seconds, ticks))))
    assert result['playback_time'] == seconds
    assert result['playback_ticks'] == ticks


def test_string_path_and_last_duplicate_field_wins(tmp_path):
    path = _demo(tmp_path, _frame(2, _timing() + _timing(2., 256)))
    assert read_demo_timing(str(path))['playback_ticks'] == 256


@pytest.mark.parametrize('frames', [b'', _frame(0), _frame(7, b'ignored')])
def test_missing_file_info(tmp_path, frames):
    with pytest.raises(ValueError, match='Missing CDemoFileInfo.*finalized'):
        read_demo_timing(_demo(tmp_path, frames))


@pytest.mark.parametrize('payload', [b'', b'\x0d' + struct.pack('<f', 10.), b'\x10\x01'])
def test_missing_timing_field(tmp_path, payload):
    with pytest.raises(ValueError, match='Missing playback_time or playback_ticks'):
        read_demo_timing(_demo(tmp_path, _frame(2, payload)))


def test_unsupported_compression(tmp_path):
    with pytest.raises(ValueError, match='Compressed CDemoFileInfo.*Snappy.*full demo parser'):
        read_demo_timing(_demo(tmp_path, _frame(2 | 64, b'unsupported')))


@pytest.mark.parametrize('cut', range(1, 16))
def test_truncated_prefix(tmp_path, cut):
    with pytest.raises(ValueError, match='Truncated demo prefix'):
        read_demo_timing(_demo(tmp_path, b'', prefix=_PREFIX[:cut]))


@pytest.mark.parametrize('frames', [
    b'\x80', b'\x02', b'\x02\x80', b'\x02\x00', b'\x02\x00\x80',
    b'\x02\x00\x05abc', b'\x07\x00\x05abc',
    _frame(2, b'\x0d\x00'), _frame(2, b'\x10\x80'),
    _frame(2, _timing() + b'\x22\x05abc'),
    _frame(2, _timing() + b'\x29\x00'),
])
def test_truncation(tmp_path, frames):
    with pytest.raises(ValueError, match='Truncated'):
        read_demo_timing(_demo(tmp_path, frames))


@pytest.mark.parametrize('bad', [b'\x80' * 5, b'\xff\xff\xff\xff\x10'])
@pytest.mark.parametrize('leading', [b'', b'\x02', b'\x02\x00'])
def test_malformed_frame_varints(tmp_path, leading, bad):
    with pytest.raises(ValueError, match='Malformed .* varint'):
        read_demo_timing(_demo(tmp_path, leading + bad))


@pytest.mark.parametrize('payload', [
    b'\x80' * 5,
    b'\x10' + b'\x80' * 10,
    b'\x10' + b'\xff' * 9 + b'\x02',
    b'\x22' + b'\x80' * 5,
])
def test_malformed_protobuf_varints(tmp_path, payload):
    with pytest.raises(ValueError, match='Malformed .* varint'):
        read_demo_timing(_demo(tmp_path, _frame(2, payload)))


@pytest.mark.parametrize('payload', [b'\x00', b'\x08\x01', b'\x15' + bytes(4)])
def test_invalid_field_keys(tmp_path, payload):
    with pytest.raises(ValueError, match='Malformed file-info'):
        read_demo_timing(_demo(tmp_path, _frame(2, payload)))


def test_unsupported_group(tmp_path):
    with pytest.raises(ValueError, match='Unsupported file-info protobuf wire type'):
        read_demo_timing(_demo(tmp_path, _frame(2, _timing() + b'\x23')))


def test_wrong_magic(tmp_path):
    with pytest.raises(ValueError, match='expected a CS2 PBDEMS2'):
        read_demo_timing(_demo(tmp_path, b'', prefix=b'HL2DEMO\0' + bytes(8)))


def test_large_payload_is_skipped_with_bounded_reads(tmp_path, monkeypatch):
    path = tmp_path / 'large.dem'
    payload_size = 8 * 1024 * 1024
    with path.open('wb') as stream:
        stream.write(_PREFIX + b'\x07\x00' + _varint(payload_size))
        stream.seek(payload_size, 1)
        stream.write(_frame(2, _timing()))

    reads = []
    seeks = []

    def tracked_open(*args, **kwargs):
        stream = builtins.open(*args, **kwargs)
        original_read = stream.read
        original_seek = stream.seek

        def read(size=-1):
            assert 0 <= size <= 16
            reads.append(size)
            return original_read(size)

        def seek(offset, whence=0):
            seeks.append((offset, whence))
            return original_seek(offset, whence)

        stream.read = read
        stream.seek = seek
        return stream

    monkeypatch.setattr(demo_timing, 'open', tracked_open, raising=False)
    assert read_demo_timing(path)['playback_ticks'] == 640
    assert sum(reads) < 100
    assert any(offset >= payload_size and whence == 0 for offset, whence in seeks)

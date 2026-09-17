"""Read CS2 timing metadata without demoparser2 or protobuf dependencies.

Format references (verified upstream):
https://raw.githubusercontent.com/SteamDatabase/Protobufs/master/csgo/demo.proto
https://raw.githubusercontent.com/LaihoE/demoparser/main/src/parser/src/first_pass/parser.rs
The latter defines the 16-byte prefix and cmd/tick/size varints, and notes
that file-info can follow DEM_Stop. Prefix offset hints are not needed here.
"""
from os import PathLike, SEEK_END
import struct
from typing import BinaryIO

_MAGIC = b'PBDEMS2\0'
_PREFIX_SIZE = 16
_FILE_INFO = 2
_COMPRESSED = 64


def _read(stream: BinaryIO, size: int, end: int, context: str) -> bytes:
    if size > end - stream.tell():
        raise ValueError(f'Truncated {context}; obtain a complete CS2 demo.')
    chunk = stream.read(size)
    if len(chunk) != size:
        raise ValueError(f'Truncated {context}; obtain a complete CS2 demo.')
    return chunk


def _varint(stream: BinaryIO, end: int, context: str, bits: int = 32) -> int:
    value = 0
    for shift in range(0, bits, 7):
        byte = _read(stream, 1, end, context)[0]
        value |= (byte & 0x7f) << shift
        if value >= 1 << bits:
            break
        if not byte & 0x80:
            return value
    raise ValueError(f'Malformed {context} varint; exceeds {bits} bits.')


def _skip(stream: BinaryIO, size: int, end: int, context: str) -> None:
    target = stream.tell() + size
    if target > end:
        raise ValueError(f'Truncated {context}; obtain a complete CS2 demo.')
    stream.seek(target)


def _file_info(stream: BinaryIO, end: int) -> dict[str, float | int | str]:
    playback_time = None
    playback_ticks = None
    while stream.tell() < end:
        key = _varint(stream, end, 'file-info field key')
        field, wire = key >> 3, key & 7
        if field == 0:
            raise ValueError('Malformed file-info: protobuf field number is zero.')
        if (field == 1 and wire != 5) or (field == 2 and wire != 0):
            raise ValueError(f'Malformed file-info: wrong wire type for timing field {field}.')
        if field == 1:
            playback_time = struct.unpack('<f', _read(stream, 4, end, 'playback_time'))[0]
        elif field == 2:
            # Protobuf int32 uses unsigned varints, sign-extended to 64 bits
            # for negative values. Decode the low 32 bits as signed.
            ticks = _varint(stream, end, 'playback_ticks', bits=64) & 0xffffffff
            playback_ticks = ticks - (1 << 32) if ticks & (1 << 31) else ticks
        elif wire == 0:
            _varint(stream, end, 'file-info value', bits=64)
        elif wire in (1, 5):
            _skip(stream, 8 if wire == 1 else 4, end, 'file-info fixed field')
        elif wire == 2:
            size = _varint(stream, end, 'file-info field length')
            _skip(stream, size, end, 'file-info length-delimited field')
        else:
            raise ValueError(f'Unsupported file-info protobuf wire type {wire}; use a full demo parser.')
    if playback_time is None or playback_ticks is None:
        raise ValueError('Missing playback_time or playback_ticks in CDemoFileInfo; obtain a finalized demo.')
    return {
        'playback_time': playback_time,
        'playback_ticks': playback_ticks,
        'timing_source': 'CDemoFileInfo',
    }


def read_demo_timing(path: str | PathLike[str]) -> dict[str, float | int | str]:
    """Return playback_time (seconds), playback_ticks and timing_source.

    Scan a seekable PBDEMS2 demo for the first uncompressed CDemoFileInfo,
    including frames after DEM_Stop. Reads are at most 16 bytes; unrelated
    packets and nested protobuf fields are skipped with seek, not allocated.

    Raise ValueError for missing/truncated/malformed timing or unsupported
    compressed file-info; filesystem errors propagate as OSError. No timing
    values or rates are validated or inferred. This is not a whole-demo
    validator: unrelated payload contents and bytes after file-info are ignored.
    Deprecated protobuf groups are unsupported; no decompressor is included.
    """
    with open(path, 'rb') as stream:
        end = stream.seek(0, SEEK_END)
        stream.seek(0)
        prefix = _read(stream, _PREFIX_SIZE, end, 'demo prefix')
        if prefix[:8] != _MAGIC:
            raise ValueError('Unsupported demo format; expected a CS2 PBDEMS2\\0 .dem file.')
        while stream.tell() < end:
            command = _varint(stream, end, 'frame command')
            _varint(stream, end, 'frame tick')
            size = _varint(stream, end, 'frame size')
            payload_end = stream.tell() + size
            if payload_end > end:
                raise ValueError('Truncated demo frame payload; obtain a complete CS2 demo.')
            if command & ~_COMPRESSED != _FILE_INFO:
                # DEM_Stop is a framed record, not the end of metadata.
                _skip(stream, size, end, 'demo frame payload')
                continue
            if command & _COMPRESSED:
                raise ValueError(
                    'Compressed CDemoFileInfo is unsupported (Snappy, flag 64); '
                    'use a full demo parser with Snappy support or provide an '
                    'uncompressed file-info record. No timing is inferred.'
                )
            return _file_info(stream, payload_end)
    raise ValueError('Missing CDemoFileInfo timing; obtain a complete, finalized CS2 demo.')

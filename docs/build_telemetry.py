"""Build the README illustration from its SVG source and a saved data example."""
import base64
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright


def data_uri(path, mime):
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:{mime};base64,{encoded}'


def trace_svg(example):
    # First encounter with a pre-hit shot, selected for a visible binary example.
    with np.load(example, allow_pickle=False) as data:
        encounters = data['x']
    index = int(np.flatnonzero(encounters[:, :, 4].sum(axis=1) > 0)[0])
    sample = encounters[index]
    left, right = 232, 856
    xs = left + np.arange(256) / 256 * (right - left)
    parts = []
    for second in range(5):
        x = left + second / 4 * (right - left)
        label = '0' if second == 4 else f'−{4 - second}'
        dash = ' stroke-dasharray="4 4"' if second == 4 else ''
        parts.append(f'<path d="M{x} 824 V1016" stroke="#e0e0e0" stroke-width="1"{dash}/>')
        parts.append(f'<text x="{x}" y="1042" font-size="16" class="muted" text-anchor="middle">{label}</text>')
    rows = [
        (0, 830, 879, -1, 6, (6, 0), '#383838'),
        (2, 898, 947, -10, 70, (60, 0), '#855536'),
        (4, 977, 1008, -.1, 1.1, (1, 0), '#383838'),
    ]
    for channel, top, bottom, low, high, labels, color in rows:
        values = sample[:, channel]
        if not (np.isfinite(values).all() and values.min() >= low and values.max() <= high):
            raise ValueError('The displayed trace needs new axis limits for this example.')
        project = lambda value: bottom - (value - low) / (high - low) * (bottom - top)
        zero = project(0)
        parts.append(f'<path d="M{left} {zero:.3f} H{right}" stroke="#d1d1d1" stroke-width="1"/>')
        for value in labels:
            parts.append(f'<text x="{left - 13}" y="{project(value) + 5:.3f}" font-size="14" class="muted" text-anchor="end">{value}</text>')
        if channel == 4:
            path = f'M{xs[0]:.3f} {project(values[0]):.3f}'
            for x, value in zip(xs[1:], values[1:]):
                path += f' H{x:.3f} V{project(value):.3f}'
        else:
            path = 'M' + ' L'.join(f'{x:.3f} {project(v):.3f}' for x, v in zip(xs, values))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round"/>')
    return '\n'.join(parts), index


def build_svg(folder):
    assets = folder / 'assets'
    faces = [
        ('Archivo', 'ArchivoRegular', 400),
        ('Archivo', 'ArchivoSemibold', 600),
        ('Newsreader Display', 'NewsreaderDisplay', 300),
    ]
    fonts = []
    for family, filename, weight in faces:
        uri = data_uri(assets / 'fonts' / f'{filename}.woff2', 'font/woff2')
        fonts.append(f"@font-face {{ font-family: '{family}'; font-weight: {weight}; font-style: normal; font-display: block; src: url('{uri}') format('woff2'); }}")
    traces, index = trace_svg(folder.parent / 'research/examples/example_validation_player.npz')
    source = (assets / 'telemetry.source.svg').read_text(encoding='utf-8')
    source = source.replace('{{FONTS}}', '\n'.join(fonts))
    source = source.replace('{{SCREENSHOT}}', data_uri(assets / 'cs2-gameplay.jpg', 'image/jpeg'))
    source = source.replace('{{TRACES}}', traces)
    (assets / 'telemetry.svg').write_text(source, encoding='utf-8')
    return index


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    index = build_svg(folder)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1120}, device_scale_factor=2)
        page.goto((folder / 'assets/telemetry.svg').as_uri(), wait_until='networkidle')
        page.evaluate('document.fonts.ready')
        page.locator('svg').screenshot(path=str(folder / 'assets/telemetry.png'))
        browser.close()
    print(f'Built telemetry.svg and telemetry.png; validation encounter index {index}.')

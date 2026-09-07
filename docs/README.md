# Report and figures

Edit `anticheat_review.source.html` for the text and `report.css` for the layout. The build produces `anticheat_review.html` with embedded fonts and figures, and `anticheat_review.pdf` for A4 printing. The HTML can be copied or opened on its own.

To rebuild the PDF from the repository root:

```bash
python -m pip install -r docs/requirements.txt
python -m playwright install chromium
python docs/build_pdf.py
```

The PDF uses Playwright 1.59 and Chromium. Newsreader and Archivo are embedded, including the Romanian characters in Ionuț’s name. Their open font licenses are included in `assets/fonts/`. Linux systems may also need Chromium’s runtime libraries, listed by Playwright’s installer.

`build_figures.py` regenerates `assets/cs2_comparison.svg` and `.png` from the frozen test metrics. Run it before `build_pdf.py` when updating the figure, and review the associated prose. The plot compares models on one benchmark, with an explicit AUC scale and confidence intervals from resampling matches.

The README uses `assets/telemetry.png`, with a portable SVG version alongside it. Edit `assets/telemetry.source.svg`, then run `python docs/build_telemetry.py` from the repository root. The build combines the unmodified Valve gameplay image with a separate aim-angle drawing and real traces from the included validation example. Sources and the exact example are documented in [illustration credits](assets/CREDITS.md).

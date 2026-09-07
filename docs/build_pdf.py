"""Build the portable HTML report and render its A4 PDF."""
import base64
from pathlib import Path
from playwright.sync_api import sync_playwright


def build_html(folder):
    faces = [
        ("Newsreader Display", "NewsreaderDisplay", 300),
        ("Newsreader Lede", "NewsreaderLede", 400),
        ("Archivo", "ArchivoRegular", 400),
        ("Archivo", "ArchivoSemibold", 600),
    ]
    fonts = []
    for family, filename, weight in faces:
        encoded = base64.b64encode((folder / "assets/fonts" / (filename + ".woff2")).read_bytes()).decode()
        fonts.append("@font-face { font-family: '" + family + "'; font-weight: " + str(weight)
                     + "; font-style: normal; font-display: block; src: url(data:font/woff2;base64,"
                     + encoded + ") format('woff2'); }")
    chart = base64.b64encode((folder / "assets/cs2_comparison.svg").read_bytes()).decode()
    source = (folder / "anticheat_review.source.html").read_text(encoding="utf-8")
    source = source.replace("{{FONTS}}", "\n".join(fonts))
    source = source.replace("{{STYLES}}", (folder / "report.css").read_text(encoding="utf-8"))
    source = source.replace("{{COMPARISON}}", "data:image/svg+xml;base64," + chart)
    (folder / "anticheat_review.html").write_text(source, encoding="utf-8")


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    build_html(folder)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto((folder / "anticheat_review.html").as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        page.pdf(path=str(folder / "anticheat_review.pdf"),
                 prefer_css_page_size=True, print_background=True)
        browser.close()
    print(folder / "anticheat_review.pdf")

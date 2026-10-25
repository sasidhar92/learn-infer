"""Wrap book/index.html (a body fragment, as published to the Claude artifact) into a
complete HTML document at docs/index.html for GitHub Pages.

    python3 book/build.py
"""
from pathlib import Path

src = Path(__file__).with_name("index.html").read_text()
out = Path(__file__).resolve().parents[1] / "docs" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(
    "<!doctype html>\n<html lang=\"en\">\n<head>\n"
    "<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
    "<meta name=\"description\" content=\"A hands-on book about why language models answer at the speed they do, measured on one laptop.\">\n"
    "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}</style>\n"
    "</head>\n<body>\n" + src + "\n</body>\n</html>\n"
)
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")

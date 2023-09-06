from pathlib import Path

import mdtex2html

out = mdtex2html.convert(Path("main.md").read_text(), extensions=["tables",
    "extra", "admonition", "codehilite", "nl2br", "fenced_code"])
Path("output.html").write_text(out)

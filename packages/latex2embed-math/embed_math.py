from __future__ import annotations

import json
import traceback
import re
from copy import copy
from tempfile import TemporaryDirectory
from pathlib import Path
from subprocess import check_output, check_call, PIPE
from multiprocessing import Lock, Pool
from multiprocessing.pool import ThreadPool
from base64 import b64encode, b64decode

from sqlitedict import SqliteDict


class NoopLock:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


####################################################################################################

# cache_path = Path("~/.cache/latex2svg_lektor/math_latex_cache.sqlite").expanduser()
cache_path = Path(__file__).parent.absolute() / "math_latex_render_cache.sqlite"
cache_path.parent.mkdir(parents=True, exist_ok=True)
encode = lambda key: b64encode(json.dumps(key).encode("utf-8"))
decode = lambda key: json.loads(b64decode(key).decode("utf-8"))
math_latex_cache = SqliteDict(cache_path, tablename="math_cache", autocommit=True)

####################################################################################################

LATEX_TEMPLATE = r"""
\documentclass[24pt]{standalone}
\usepackage{amsmath,amssymb}
\begin{document}

$REPLACE_MATH$

\end{document}
"""


def convert_with_latex(dirpath: str | Path, math_text: str) -> str:
    dirpath = Path(dirpath)
    (dirpath / "main.tex").write_text(LATEX_TEMPLATE.replace("REPLACE_MATH", math_text))
    check_call(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            f"-output-directory={str(dirpath)}",
            "main.tex",
        ],
        stdout=PIPE,
        stderr=PIPE,
    )
    output = check_output(
        ["inkscape", str(dirpath / "main.pdf"), "--export-type=svg", "--export-filename=-"],
        stderr=PIPE,
    ).decode("utf-8")
    output = (dirpath / "main.svg").read_text()
    matches = list(re.findall(r"<svg.*?</svg>", output, re.DOTALL))
    assert len(matches) >= 1
    output = matches[0]
    return output


def convert_with_mathjax(dirpath: str | Path, math_text: str) -> str:
    dirpath = Path(dirpath)
    input_file = dirpath / "main.txt"
    input_file.write_text(math_text)
    output_file_no_ext = dirpath / "main"
    main_js = Path(__file__).parent / "mathjax_converter" / "main.js"
    check_call(
        ["node", main_js, input_file, output_file_no_ext]
        # stderr=PIPE,
        # stdout=PIPE,
    )
    output = output_file_no_ext.with_suffix(".mml").read_text()
    return output


def fix_latex_encoding(text: str) -> str:
    text = re.sub(r"([^\\])\\\s*(\n| )", r"\1\\\\\2", text, re.DOTALL)
    text = re.sub(r"&amp;", "&", text, re.DOTALL)
    text = re.sub(r"<em>", "_", text, re.DOTALL)
    text = re.sub(r"</em>", "_", text, re.DOTALL)
    text = re.sub(r"<b>", "_", text, re.DOTALL)
    text = re.sub(r"</b>", "_", text, re.DOTALL)
    return text


####################################################################################################


def remove_backticked_text(text) -> tuple[str, dict[str, str]]:
    idx = 0
    replace_map = {}
    while True:
        its = re.finditer(r"```.*?```", text, flags=re.DOTALL)
        try:
            it = next(its)
        except StopIteration:
            break
        key = f"REPLACE_PATTERN_{idx:020d}"
        idx += 1
        replace_map[key] = text[it.start() : it.end()]
        text = text[: it.start()] + key + text[it.end() :]
    while True:
        its = re.finditer(r"`.*?`", text, flags=re.DOTALL)
        try:
            it = next(its)
        except StopIteration:
            break
        key = f"REPLACE_PATTERN_{idx:020d}"
        idx += 1
        replace_map[key] = copy(text[it.start() : it.end()])
        text = text[: it.start()] + key + text[it.end() :]
    return text, replace_map


def restore_backticked_text(text, replace_map):
    for key, value in replace_map.items():
        text = text.replace(key, value)
    return text


####################################################################################################


def convert_math(math_text: str, lock: Lock | None = None):
    math_text = fix_latex_encoding(math_text)
    lock = lock if lock is not None else NoopLock()
    with lock:
        if encode(math_text) in math_latex_cache:
            return decode(math_latex_cache[encode(math_text)])
    output = f" <span style='color:red'> Conversion of {math_text} failed </ span> "
    with TemporaryDirectory() as dir:
        try:
            dirpath = Path(dir).absolute()
            output = convert_with_mathjax(dirpath, math_text)
        except Exception:
            traceback.print_exc()
    with lock:
        math_latex_cache[encode(math_text)] = encode(output)
        math_latex_cache.commit()
    return output


####################################################################################################


def latex2embed_math_render(document_text: str) -> str:
    document_text, replace_map = remove_backticked_text(document_text)
    # convert all top cache
    its = list(re.finditer(r"\$\$(.*?)\$\$", document_text, flags=re.DOTALL)) + list(
        re.finditer(r"[^\$]\$(.*?)\$[^\$]", document_text, flags=re.DOTALL)
    )
    lock = Lock()
    with ThreadPool() as pool:
        pool.starmap(convert_math, [(it.groups()[0], lock) for it in its])
    # for it in its:
    #    convert_math(it.groups()[0])

    # centered math #######################################
    while True:
        its = re.finditer(r"\$\$(.*?)\$\$", document_text, flags=re.DOTALL)
        try:
            match = next(its)
        except StopIteration:
            break
        math_text = match.groups()[0]
        new_html_text = convert_math(math_text)
        document_text = (
            document_text[: match.start()]
            + " <p align='center'>"
            + re.sub(r"\s+", " ", new_html_text.strip(), flags=re.DOTALL)
            + "</p> "
            + document_text[match.end() :]
        )
    # inline math #########################################
    while True:
        its = re.finditer(r"[^\$]\$(.*?)\$[^\$]", document_text, re.DOTALL)
        try:
            match = next(its)
        except StopIteration:
            break
        math_text = match.groups()[0]
        new_html_text = convert_math(math_text)
        document_text = (
            document_text[: match.start() + 1].strip()
            + " <span>"
            + re.sub(r"\s+", " ", new_html_text.strip(), flags=re.DOTALL)
            + "</span> "
            + document_text[match.end() - 1 :].strip()
        )
    return restore_backticked_text(document_text, replace_map)


####################################################################################################
if __name__ == "__main__":
    document_text = Path("main.html").read_text()
    document_text = latex2embed_math_render(document_text)
    Path("./main2.html").write_text(document_text)

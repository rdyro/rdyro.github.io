# -*- coding: utf-8 -*-
from __future__ import annotations

from lektor.pluginsystem import Plugin
from lektor.markdown import Markdown

from copy import deepcopy
import sys
from pathlib import Path

try:
    from .embed_math import latex2embed_math_render
except (ImportError, ModuleNotFoundError):
    sys.path.append(str(Path(__file__).absolute().parent))
    from embed_math import latex2embed_math_render


def convert_fn(text: str | Markdown) -> str | Markdown:
    if isinstance(text, Markdown):
        ret = deepcopy(text)
        ret.source = latex2embed_math_render(text.source)
        return ret
    else:
        return latex2embed_math_render(text)


class Latex2EmbedMathPlugin(Plugin):
    name = "latex2embed_math"
    description = (
        "Filters a partially rendered Markdown, replacing all math with static"
        + " embedded HTML math using mathjax."
    )

    def on_setup_env(self, **extra):
        self.env.jinja_env.filters.update(latex2embed_math=convert_fn)

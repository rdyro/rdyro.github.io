import ast
import io
import re

from setuptools import setup, find_packages

with io.open("README.md", "rt", encoding="utf8") as f:
    readme = f.read()

description = (
    "Filters a partially rendered Markdown, replacing all math with static"
    + " embedded HTML math using mathjax."
)

setup(
    author="Robert Dyro",
    author_email="rdyro@stanford.edu",
    description=description,
    keywords="Lektor plugin",
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    name="lektor-latex2embed-math",
    packages=find_packages(),
    py_modules=["lektor_latex2embed_math"],
    # url='[link to your repository]',
    version="0.1",
    classifiers=[
        "Framework :: Lektor",
        "Environment :: Plugins",
    ],
    entry_points={
        "lektor.plugins": [
            "latex2embed-math = lektor_latex2embed_math:Latex2EmbedMathPlugin",
        ]
    },
)
